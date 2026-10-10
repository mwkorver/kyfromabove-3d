#!/usr/bin/env python3
"""Estimate a camera pose for every frame of a video with LingBot-Map.

The other end of ``score_trajectory.py``: this makes the estimate it scores.
LingBot-Map (github.com/Robbyant/lingbot-map) is a streaming video model that
gives each frame a camera (pose and focal length) and a depth map, in its
own units, with no north, scale or origin. Its own ``batch_demo.py`` needs an
NVIDIA GPU for its renderer; this runs only the model, on CUDA, Apple's GPU
(MPS) or the CPU, and writes what ``score_trajectory.py`` reads:

    poses/frame_000000.npz ...  extrinsic (3x4, camera to world, OpenCV axes),
                                intrinsic (3x3, at the model's input size),
                                depth, world_points and their confidence,
                                whichever the model returns (float16)
    poses/meta.json             frame files, input size, timings, settings

Run it with LingBot-Map's own Python, not this project's: it needs that
repository on the path and torch, which this project does not install::

    ~/working/lingbot-map/.venv/bin/python scripts/video_poses.py \\
        --frames data/derived/video/DX0001_003049/frames \\
        --model ~/working/lingbot-map/checkpoints/lingbot-map.pt \\
        --out data/derived/video/DX0001_003049/poses
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def invert_3x4(m: np.ndarray) -> np.ndarray:
    """Invert rigid [R | t] matrices, shape (..., 3, 4)."""
    rt = np.swapaxes(m[..., :3], -1, -2)
    return np.concatenate([rt, -rt @ m[..., 3:]], axis=-1)


def load_lean(model_args, device, torch):
    """Build the model and load its weights without holding them twice.

    LingBot-Map's ``demo.load_model`` reads the 4.6 GB checkpoint onto the
    device and builds the model in float32 beside it, about 11 GB at its
    peak: more than a 16 GB Mac has to spare. Here the checkpoint is mapped
    from disk, the trunk (the aggregator, nearly all the weights) is copied
    in at half precision as ``demo.py`` does on a CUDA GPU, and the heads stay
    float32. Returns the model and the dtype to run the trunk in.
    """
    from demo import load_model

    weights, model_args.model_path = model_args.model_path, None
    model = load_model(model_args, torch.device("cpu"))
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    elif device.type == "mps":
        dtype = torch.float16
        # The model switches autocast off around its heads with
        # autocast("cuda", enabled=False); make that apply on Apple's GPU too.
        real = torch.amp.autocast
        torch.amp.autocast = lambda device_type, *a, **k: real(
            "mps" if device_type == "cuda" else device_type, *a, **k)
    if dtype != torch.float32:
        model.aggregator = model.aggregator.to(dtype)
    ckpt = torch.load(weights, map_location="cpu", mmap=True, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    del ckpt
    print(f"weights loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")
    if device.type == "mps":
        # The 3D rotary position table is complex128; Apple's GPU has no
        # 64-bit floats, so keep it in complex64 (its angles need no more).
        for m in model.modules():
            f = getattr(m, "freqs", None)
            if isinstance(f, torch.Tensor) and f.dtype == torch.complex128:
                m.freqs = f.to(torch.complex64)
    return model.to(device).eval(), dtype


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True, help="folder of frames, in order by name")
    ap.add_argument("--model", required=True, help="lingbot-map.pt")
    ap.add_argument("--repo", default=os.path.expanduser("~/working/lingbot-map"),
                    help="LingBot-Map checkout (default ~/working/lingbot-map)")
    ap.add_argument("--out", required=True, help="folder for frame_*.npz and meta.json")
    ap.add_argument("--first", type=int, default=None, help="only the first N frames (a speed test)")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--keyframe-interval", type=int, default=None,
                    help="every Nth frame is kept in the model's memory (default: 1 up to 320 frames)")
    ap.add_argument("--scale-frames", type=int, default=8,
                    help="frames the model looks at together first to set its scale")
    ap.add_argument("--kv-window", type=int, default=16,
                    help="keyframes the model keeps in memory besides the scale frames (LingBot-Map's "
                         "default is 64, about 5 GB at 518x294: too much beside a 16 GB Mac's other work)")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: the best there is)")
    args = ap.parse_args(argv)

    sys.path.insert(0, args.repo)
    import torch
    from demo import postprocess
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    paths = sorted(glob.glob(os.path.join(args.frames, "*.jpg")) + glob.glob(os.path.join(args.frames, "*.png")))
    paths = paths[:: args.stride]
    if args.first:
        paths = paths[: args.first]
    if not paths:
        print(f"no frames in {args.frames}", file=sys.stderr)
        return 1
    n = len(paths)
    kf = args.keyframe_interval or (1 if n <= 320 else (n + 319) // 320)

    t0 = time.time()
    images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    h, w = images.shape[-2:]
    model_args = SimpleNamespace(
        mode="streaming", image_size=518, patch_size=14, enable_3d_rope=True,
        max_frame_num=1024, kv_cache_sliding_window=args.kv_window, num_scale_frames=args.scale_frames,
        use_sdpa=device.type != "cuda", camera_num_iterations=4, model_path=args.model,
    )
    model, dtype = load_lean(model_args, device, torch)
    t_load = time.time() - t0
    print(f"{n} frames at {w}x{h} on {device}, keyframe interval {kf}; loaded in {t_load:.0f} s", flush=True)

    t0 = time.time()
    amp = torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else torch.no_grad()
    with torch.no_grad(), amp:
        pred = model.inference_streaming(
            images.to(device), num_scale_frames=args.scale_frames,
            keyframe_interval=kf, output_device=torch.device("cpu"),
        )
    t_inf = time.time() - t0
    print(f"inference {t_inf:.0f} s ({t_inf / n:.2f} s a frame)", flush=True)
    pred, _ = postprocess(pred, pred.get("images", images))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("frame_*.npz"):
        old.unlink()
    ext, intr = pred["extrinsic"].float().numpy(), pred["intrinsic"].float().numpy()
    # demo.postprocess says it turns the model's world-to-camera into camera
    # to world, but on DX0001_003049 (a drone flying toward downtown, rising
    # and pitching down) its output reads as backing away, sinking and
    # pitching up: it is world to camera. Invert it so the files hold what
    # the docstring says.
    ext = invert_3x4(ext)
    dense = {k: pred[k].float().numpy().astype(np.float16)
             for k in ("depth", "depth_conf", "world_points", "world_points_conf") if k in pred}
    for i in range(n):
        np.savez(out / f"frame_{i:06d}.npz", extrinsic=ext[i], intrinsic=intr[i],
                 **{k: v[i] for k, v in dense.items()})
    meta = {
        "frames": [os.path.relpath(p, out) for p in paths], "input_size": [int(w), int(h)],
        "extrinsic": "camera to world, OpenCV axes (x right, y down, z forward)",
        "dense": sorted(dense), "device": str(device), "keyframe_interval": kf, "kv_window": args.kv_window, "scale_frames": args.scale_frames,
        "stride": args.stride, "load_s": round(t_load, 1), "inference_s": round(t_inf, 1),
        "model": os.path.basename(args.model),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {n} frames to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
