# Phase 1 — Local camera and imagery validation: walkthrough

## What changed and why

The previous walkthrough reported **0.000 px** median check-feature error across
35 observations and passed every acceptance criterion. That number was not a
measurement. `scripts/validate_building.py` computed each "observed" image
position by projecting the check feature through the camera under test, then
`qa.add_observation` projected the same point again and differenced the two. The
difference is zero as an algebraic identity — it would have printed 0.000 px
with the focal length halved or the rotation transposed.

Rebuilding the QA around a real measurement immediately found a real bug.

---

## 1. The bug the old QA could not see

`cameras.py` composed the photogrammetric rotation in the wrong order:

```
was:  R_cw = diag(1,−1,−1) · Rx(−ω) · Ry(−φ) · Rz(−κ)
now:  R_cw = diag(1,−1,−1) · (Rx(ω) · Ry(φ) · Rz(κ))ᵀ
            = diag(1,−1,−1) · Rz(−κ) · Ry(−φ) · Rx(−ω)
```

Same three rotations, wrong order. At ω ≈ 45° that is worth **136 px of absolute
projection error, about 9.4 m on the ground**.

| | before | after |
|---|---|---|
| Median cross-view residual | 26.34 px | **0.92 px** |
| Systematic component | 163 cm | **2.2 cm** |

It survived the original 11 tests because every one was a self-consistency check
— orthonormality, det(R) = 1, unproject/project round-trip — and those pass for
any rotation matrix whatsoever. It also survived visual review: 136 px on a
1200 px crop still looks broadly aligned in a LiDAR overlay, which is exactly
how it got waved through.

---

## 2. How reprojection is measured now

For each LiDAR anchor point, the image content around it is resampled onto the
anchor's locally-fitted surface plane in two different exposures. Rectifying
both views through a common plane removes the perspective difference between
them, so if the camera model is right the two patches are the same picture.
Normalized cross-correlation finds the offset.

The imagery decides the result. No number is derived from the model being
tested.

- **Anchors**: LiDAR points on locally planar surfaces (plane-fit RMS ≤ 8 cm,
  near-horizontal normal), chosen on an 8 m grid so they span the building.
- **Check features**: roof corners derived from the LiDAR roof surface via
  minimum-area rectangle. The old ones were seven hardcoded literals with no
  provenance that plotted into the parking lot.
- **Second, independent check**: correlation-measured image positions are
  triangulated and compared against LiDAR, which catches errors common to every
  exposure that a relative metric cannot see.

### Proof it is sensitive

Perturb one exposure, re-measure, on the real data:

| perturbation | median residual |
|---|---|
| none (baseline) | 0.82 px |
| perspective center +0.25 m | 3.91 px |
| perspective center +1.0 m | 14.23 px |
| ω +0.01° | 6.12 px |
| ω +0.05° | 29.07 px |

The old metric returns 0.00 px for every row.

---

## 3. From 3 exposures to 18

Frame selection used to test whether an image footprint polygon covered the AOI
*center point*. A frame can cover that point and still have the building running
off the top of the sensor. Coverage is now decided by projecting the eight
building-volume corners.

On the Fwd camera that rejects two of five:

| exposure | corners on sensor | margin | |
|---|---|---|---|
| Fwd_2034_201822 | 4 / 8 | −249 px | **rejected** |
| Fwd_2034_201823 | 8 / 8 | 1730 px | keep |
| Fwd_2034_201824 | 8 / 8 | 3739 px | keep |
| Fwd_2034_201825 | 8 / 8 | 2832 px | keep |
| Fwd_2034_201826 | 4 / 8 | −482 px | **rejected** |

Three exposures inside a 4.7° azimuth cone is not enough to reconstruct from.
The other camera heads fly *different flight lines* over the same ground, so the
config no longer pins a single flight line or look direction:

```yaml
flight_ids: null
camera_directions: ["Fwd", "Left", "Right", "Bwd", "Color"]
```

That gives **18 usable exposures**, all 18 fully containing the building:

| camera | line | exposures |
|---|---|---|
| Bwd | 2034 | 4 |
| Color (nadir) | 2034 | 4 |
| Fwd | 2034 | 3 |
| Left | 2031 | 3 |
| Left | 2036 | 2 |
| Right | 2032 | 2 |

The Color head is a different sensor — 20544 × 14016 at 79.60 mm, against
14144 × 10560 at 123.38 mm for the obliques — read straight from the EO record,
and it looks straight down (ω, φ, κ all ≈ 0).

| | Fwd only | 4 oblique heads | + Color |
|---|---|---|---|
| Max view-ray separation | 8.9° | 103.3° | **103.3°** |
| Azimuth concentration R | 1.00 | 0.17 | **0.11** |
| Elevation range | 39–48° | 36–52° | **36–81°** |
| Median views per triangulated point | 3 | 5 | 5 |
| Median max ray angle | 8.9° | 17.7° | **44.3°** |
| Median 3D discrepancy vs LiDAR | 0.70 m | 0.47 m | **0.28 m** |
| Vertical bias vs LiDAR | −0.29 m | +0.02 m | **+0.01 m** |

The vertical bias collapsing to +0.01 m shows the original figure was weak
intersection geometry, not a datum error. The nadir head more than doubles the
ray intersection angle, halving the 3D discrepancy again.

A detail worth noting: azimuth is arbitrary for a view looking straight down, so
`view_geometry_summary` weights each bearing by cos(elevation). A nadir exposure
contributes nothing to the concentration figure rather than swinging it.

Crops are sized from the widest projection — 1600 px now, not a hardcoded 1200.

## 3a. A second calibration bug, found the same way

The Left and Right heads carry `CamPpyMm = 6.68`, a ~1776 px principal-point
offset whose sign the documentation does not state. The original code added it.
It must be **subtracted**: `CamPpyMm` is a photo-frame offset with +y up, while
image rows increase downward. Getting it backwards moves the principal point by
3552 px.

Settled by measurement, not by argument — matching against a validated Fwd
exposure:

| camera | with `+` | with `−` |
|---|---|---|
| Left_2036_200821 | 0 / 447 matches | 222 / 447, median 6.3 px |
| Right_2032_300787 | 6 / 447, median 27.6 px | 213 / 447, median 3.3 px |
| Bwd_2034_201836 (offset 0) | 190 / 447, 2.28 px | 190 / 447, 2.28 px |

Bwd carries no offset and is identical either way, which makes it a clean
control on the experiment.

## 4. Other corrections

- **Validity masks.** Were uniformly 255 on all five crops — the mask flagged any
  non-black pixel as valid, which on a clamped window is every pixel. Windows are
  now padded rather than clamped and the mask records real sensor coverage. They
  are uniform 255 again today, but now because coverage genuinely is 100%; the
  off-sensor path has a test.
- **Provenance.** `source_manifest.json` shipped `"assets": {}` behind a
  provenance PASS — notebook 01 had a `# Add primary source assets` comment
  followed by no `add_asset` calls. Registration now goes through one function,
  `build_source_manifest`, and records size and SHA-256 for all three inputs.
- **"Undistorted" crops.** The EO table carries no distortion coefficients, so
  nothing was ever undistorted. The crops are pinhole windows onto imagery the
  vendor has already rectified; the docs now say that.
- **Flight date.** Was recorded as Nov 20, 2022. Both GeoPackages say
  **2022-11-21**, 12:59 EST.

---

## 5. Results

Two groupings, because they answer different questions. **Within group** means
consecutive pairs sharing a camera head and flight line — like compared with
like, so it tests the camera model. **Across look directions** additionally
carries line-to-line georeferencing bias.

| group | exposures | median |
|---|---|---|
| Bwd, line 2034 | 4 | 0.85 px |
| Color, line 2034 | 4 | 0.94 px |
| Fwd, line 2034 | 3 | 0.92 px |
| Left, line 2031 | 3 | 1.17 px |
| Left, line 2036 | 2 | 1.87 px |
| Right, line 2032 | 2 | 1.26 px |
| **combined within-group** | **18** | **1.04 px** (p90 2.95, 99% accepted) |
| across look directions | 18 | 1.54 px (2,749 matches) |

Every camera head is individually well modeled — including Color, whose
different focal length and sensor size come straight from the EO record. That is
what confirms the principal-point sign fix.

Cross-direction, by head: Fwd 1.06, Color 1.35, Left 2.92, Bwd 2.98, Right
4.06 px. Color ties in better than the side heads because nadir-against-oblique
is a smaller view-angle change than oblique-against-opposing-oblique.

| criterion | target | measured | status |
|---|---|---|---|
| Usable exposures | ≥ 3 | 18 | **PASS** |
| Within-group residual | < 2.0 px | **1.04 px** over 5,304 matches | **PASS** |
| Match acceptance rate | ≥ 80% | 99% | **PASS** |
| View-ray separation | ≥ 30° | 103.3°, 5 camera heads | **PASS** |
| Every crop contains the building | all | 18 / 18 | **PASS** |
| Crop padding on sensor | ≥ 75% | min 0.807 | **PASS** |
| Source provenance with checksums | all | 3 with SHA-256 | **PASS** |

Unit tests: **24 passed**, including a synthetic-scene suite that renders two
views, verifies a self-match measures zero, and confirms injected translations
and rotations are recovered — plus tests pinning the rotation convention and the
principal-point sign so neither can silently flip back.

## 6. What these numbers do not say

- **Residuals are relative.** They measure agreement between exposures and with
  LiDAR. Absolute georeferencing accuracy is not established; that needs
  surveyed ground control, which this dataset does not have.
- **Cross-direction ties are looser than within-line ones** — 1.54 px against
  1.04 px, and 4.06 px for the Right head. Per frame that is 0.1–2.9 px of
  systematic offset plus 1–4 px of scatter. Phase 2 should plan on a bundle
  adjustment across look directions rather than treating the delivered EO as
  final.
- **One crop is padded at the sensor edge.** `Bwd_2034_201839` holds the whole
  building (8/8 corners, 135 px margin) but 19% of its 1600 px window falls off
  the sensor. The validity mask records it; anything that uses the crop should honor the mask.
- **All five heads over this AOI are now used.** Nothing further is available
  on this collection without relaxing the containment requirement.

## 7. Working with this in Jupyter

```bash
jupyter lab
```

| notebook | contents |
|---|---|
| `01_inspect_sources.ipynb` | calibration, provenance, LiDAR footprint, frame coverage, view geometry |
| `02_validate_cameras.ipynb` | anchors, within-group and cross-direction residuals, sensitivity table, triangulation |
| `03_validate_crops.ipynb` | crops, masks, handoff package |

All three are checked in with executed outputs. `Project()` locates the
repository root itself, so notebooks run from any working directory:

```python
from kyfromabove_3d import Project, build_planar_anchors, readers_for_cameras
from kyfromabove_3d.qa import (measure_cross_view_residuals,
                          measure_within_group_residuals)

proj = Project()
anchors = build_planar_anchors(proj.lidar)
readers = readers_for_cameras(proj.cameras)

# within group: same camera head and flight line, so this tests the model
measure_within_group_residuals(anchors, proj.cameras, readers)

# across look directions: pick the reference explicitly
measure_cross_view_residuals(anchors, proj.cameras, readers,
                             pair_mode="reference", reference="Fwd_2034_201824")
```

`ReprojectionReport` has `_repr_html_`, `.to_dataframe()`, `.stats()` and
`.save_json()`. Every function in `kyfromabove_3d.plots` returns a Figure so it
renders inline, and takes an optional `save_to`.

Batch equivalent:

```bash
python scripts/validate_building.py
pytest -q
```
