"""Planes fitted to the LiDAR: tangent bases, local plane fits, how coherent a surface is."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------- planes
def tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane with the given normal."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    # Cross with whichever world axis is least parallel to n.
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(n)))] = 1.0
    e1 = np.cross(n, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _outward_normal(xyz: np.ndarray, index: int, normal: np.ndarray,
                    slab_m: float = 3.0, radius_m: float = 40.0) -> np.ndarray:
    """Flip a plane normal so it points out of the building, not into it.

    ``fit_local_planes`` returns an arbitrary sign, and for a facade the sign is
    load-bearing: it says which side of the wall a camera must be on to see the
    surface. On 400 West Market, 68% of anchor/camera pairs sit behind the wall,
    and attempting them yields guaranteed ``low_ncc`` rejections that read as a
    failure of the method rather than of the geometry.

    Orientation is against the centroid of the building's **cross-section at
    that height** -- points in a thin z-slab, within ``radius_m`` horizontally.
    A closed cross-section has its centroid inside it, so outward is away.

    Two simpler tests were tried and are wrong. A centroid of the local
    neighbourhood sits *in* the wall plane, because on a hollow cloud the
    neighbourhood is the wall. An occupancy probe -- "LiDAR never sees inside a
    building, so the emptier side is outside" -- is false: the outside of a wall
    is air and is equally empty. On a clean hollow box it scored 60%, barely
    above chance.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    p = xyz[index]
    slab = xyz[(np.abs(xyz[:, 2] - p[2]) <= slab_m)
               & (np.linalg.norm(xyz[:, :2] - p[:2], axis=1) <= radius_m)]
    if len(slab) < 8:
        return n
    inward = np.median(slab[:, :2], axis=0) - p[:2]
    if np.linalg.norm(inward) < 1e-9:
        return n
    return -n if np.dot(n[:2], inward) > 0 else n


def facade_tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tangent basis for a near-vertical plane: e1 along the wall, e2 up it.

    :func:`tangent_basis` picks whichever world axis is least parallel to the
    normal, which is numerically fine but leaves the in-plane axes arbitrary.
    NCC does not care, but the reported ``offset_e1_m`` and ``offset_e2_m`` do:
    on a facade they should read as "along the wall" and "up the wall", not as
    two unnamed directions. Falls back when the plane is not actually vertical.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), n)
    norm = np.linalg.norm(e1)
    if norm < 1e-6:                      # horizontal plane: no "along the wall"
        return tangent_basis(n)
    e1 /= norm
    e2 = np.cross(n, e1)
    return e1, e2 / np.linalg.norm(e2)


def fit_local_planes(
    xyz: np.ndarray, k: int = 16, max_radius_m: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a plane to each point's k nearest neighbours.

    Returns ``(normals, rms_residual_m, n_used)``. Points whose neighbourhood is
    sparser than ``max_radius_m`` get a large residual so they are filtered out
    downstream rather than silently producing a garbage normal.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    tree = cKDTree(xyz)
    dists, idx = tree.query(xyz, k=min(k, len(xyz)))

    normals = np.zeros_like(xyz)
    residuals = np.full(len(xyz), np.inf)
    n_used = np.zeros(len(xyz), dtype=int)

    for i in range(len(xyz)):
        sel = idx[i][dists[i] <= max_radius_m]
        if len(sel) < 5:
            continue
        pts = xyz[sel]
        centroid = pts.mean(axis=0)
        centred = pts - centroid
        # Smallest singular vector is the plane normal.
        _, sv, vh = np.linalg.svd(centred, full_matrices=False)
        normal = vh[2]
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal
        residuals[i] = float(np.sqrt(np.mean((centred @ normal) ** 2)))
        n_used[i] = len(sel)

    return normals, residuals, n_used


def surface_coherence(
    xyz: np.ndarray,
    outer_k: int = 48,
    inner_k: int = 8,
    max_radius_m: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate real structure from range noise in a rough surface.

    Returns ``(coherence, offset_m)``. A high local plane residual means a point
    does not sit on a plane, but says nothing about *why*: a roof vent and a
    noisy grazing-incidence facade return look identical to
    :func:`fit_local_planes`. On 400 West Market two thirds of the surface has a
    residual above 0.08 m, and densifying noise would resolve nothing while
    looking like progress.

    The distinguishing property is **spatial coherence**. Real structure makes a
    point's immediate neighbours deviate from the surrounding surface *together*
    and in the same direction -- a step, a ridge, a parapet, a mechanical unit.
    Range noise makes them deviate independently.

    So a plane is fitted to a wide neighbourhood (``outer_k``) and the signed
    deviations of a tight one (``inner_k``) are measured against it:

        coherence = |mean(d)| / mean(|d|)


    which is 1.0 when the inner cluster sits wholly to one side of the
    surrounding surface, and near 0 when it scatters. ``offset_m`` is
    ``mean(d)`` -- how far, and which way.

    Deliberately not a smoothness measure. A rough *but coherent* surface is
    detail worth resolving; a rough incoherent one is measurement error.

    **``outer_k`` sets the largest structure this can see.** The wide
    neighbourhood has to reach *past* a feature for the feature to deviate from
    it; anything bigger becomes the local surface and reports no offset. At the
    pilot's in-volume density of 2.82 pts/m^2 the default spans roughly a 2.3 m
    radius, which covers vents, parapets, ledges and roof steps. A whole roof
    level is invisible to it, and should be — that is what
    :func:`building_surface` is for.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    coherence = np.zeros(n)
    offset = np.zeros(n)
    if n < outer_k:
        return coherence, offset

    tree = cKDTree(xyz)
    dists, idx = tree.query(xyz, k=min(outer_k, n))

    for i in range(n):
        near = idx[i][dists[i] <= max_radius_m]
        if len(near) < max(inner_k + 4, 12):
            continue
        pts = xyz[near]
        centroid = pts.mean(axis=0)
        centred = pts - centroid
        normal = np.linalg.svd(centred, full_matrices=False)[2][2]
        # SVD returns an arbitrary sign, which would make the offset's direction
        # meaningless. Orient upward so positive reads as "above the local
        # surface" -- a vent rather than a light well.
        if normal[2] < 0:
            normal = -normal

        # Signed deviation of the tight neighbourhood from the wide plane.
        inner = idx[i][:inner_k]
        d = (xyz[inner] - centroid) @ normal
        denom = float(np.mean(np.abs(d)))
        if denom < 1e-9:
            continue
        coherence[i] = abs(float(np.mean(d))) / denom
        offset[i] = float(np.mean(d))

    return coherence, offset
