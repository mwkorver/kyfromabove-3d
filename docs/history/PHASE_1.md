# Phase 1 — Local camera and imagery validation: verification report

## Summary

Kentucky From Above oblique imagery, the delivered exterior orientation, and the
Phase 1 LiDAR do produce geometrically consistent building crops — but only
after correcting a rotation-convention error in the camera model, and only for
three of the five exposures originally shipped.

**Measured result: median within-group reprojection residual 1.02 px** over
5,329 accepted matches on 18 exposures spanning five camera heads
(target < 2.0 px).

This supersedes an earlier report that claimed 0.000 px over 35 observations.
That figure was an artifact: the QA computed each "observed" image position by
projecting the check feature through the camera it was testing, then differenced
that against a second projection of the same point. The result was zero by
construction and independent of whether the camera model was correct.

---

## 0. Data source

All imagery and elevation data is from [KyFromAbove](https://kyfromabove.ky.gov/),
the Commonwealth of Kentucky's aerial photography and elevation program,
distributed public-domain via the AWS Registry of Open Data. Kentucky publishes
oblique frames *with* their exterior orientation tables, which almost no other
U.S. state does, and which is what makes this work possible at all.

## 1. Pilot dataset

| | |
|---|---|
| AOI | Grayson, Carter County, Kentucky |
| Building | `building_001` — Tractor Supply Co. |
| Horizontal CRS | EPSG:3089 (Kentucky Single Zone), NAD83, US survey feet |
| Vertical datum | NAVD88, US survey feet |
| Scale | s = 1200/3937 ≈ 0.3048006096012192 m per US survey foot (exact) |
| Local origin | (5725184.60, 4025570.90, 600.00) ft → (0, 0, 0) m |
| Flight lines | 2031, 2032, 2034, 2036 — all flown **2022-11-21** |
| Cameras | `434S82525X818326-f120_v2`: four oblique heads (Fwd, Bwd, Left, Right) at f = 123.38 mm, 14144 × 10560 px, and the nadir Color head at f = 79.60 mm, 20544 × 14016 px. All 3.76 µm pitch. |
| LiDAR | `s3://kyfromabove/elevation/PointCloud/Phase2/N067E390_LAS_Phase2.copz.laz`, 37,071 points in AOI |

Building footprint measured from the LiDAR roof surface (3,157 returns):
**38.8 × 49.2 m ≈ 1,908 m²**.

### Interior orientation

The EO table supplies focal length, pixel pitch and a principal-point offset.
It supplies **no distortion coefficients**. Nothing in this pipeline undistorts
anything; the crops are plain pinhole windows onto imagery the vendor has
already rectified. Earlier text describing "undistorted crops" overstated what
the code does.

`CamPpyMm = 6.68` on the Left and Right cameras — a ~1776 px principal-point
offset whose sign the documentation does not state. It is **subtracted**:
`CamPpyMm` is a photo-frame offset with +y up, while image rows increase
downward. Getting it backwards displaces the principal point by 3552 px.

The sign was settled by measurement. Matching a Left and a Right exposure
against a validated Fwd exposure accepts 0 and 6 correspondences with `+`, and
222 and 213 with `-`. The Bwd camera carries no offset and is unaffected either
way, which makes it a clean control.

---

## 2. Camera model

### Rotation

    M_photo_to_ground = Rx(ω) · Ry(φ) · Rz(κ)
    R_cw              = diag(1, −1, −1) · M_photo_to_groundᵀ
                      = diag(1, −1, −1) · Rz(−κ) · Ry(−φ) · Rx(−ω)

**This was previously implemented as `Rx(−ω)·Ry(−φ)·Rz(−κ)`** — the same three
rotations composed in the wrong order. With ω ≈ 45° the error is not small:

| | wrong order | correct order |
|---|---|---|
| Absolute projection shift | 136 px (≈ 9.4 m on the ground) | — |
| Median cross-view residual | 26.34 px | **0.92 px** |
| Systematic component | 163 cm | **2.2 cm** |

The error survived the original test suite because every test was a
self-consistency check — orthonormality, det(R) = 1, unproject/project
round-trip — and all of those pass for *any* rotation matrix. It also survived
visual inspection: a 136 px shift on a 1200 px crop still looks broadly aligned
in a LiDAR overlay.

### Projection

    x_c = R_cw (X_local − C_local),   u = f_x·x_c/z_c + c_x,   v = f_y·y_c/z_c + c_y

### Crop intrinsics

    c_x' = c_x − x₀,  c_y' = c_y − y₀,  f_x' = f_x,  f_y' = f_y

---

## 3. What is measured, and how

### Cross-view reprojection residual (primary)

For each LiDAR anchor, the image content around it is resampled onto the
anchor's locally-fitted surface plane in two exposures. Rectifying through a
common plane removes the perspective difference between views, so a correct
camera model makes the two patches the same picture. Normalized
cross-correlation locates the offset; the imagery decides the answer.

Anchors are LiDAR points on locally planar surfaces (plane fit RMS ≤ 8 cm,
near-horizontal normal), selected on an 8 m grid so they span the building
rather than clustering where returns are densest.

Two groupings are reported, because they answer different questions:

- **Within group** — consecutive pairs sharing a camera head *and* a flight
  line. Like compared with like, so this tests the camera model. It is the
  gated criterion.
- **Across look directions** — a validated Fwd exposure against everything
  else. This additionally carries line-to-line georeferencing bias, and
  correlating two views 100° apart is genuinely harder, so it is reported
  rather than gated.

### Triangulation against LiDAR (secondary)

Correlation-measured image positions are triangulated and compared with the
LiDAR position of the same point. A relative metric cannot see an error common
to every exposure; this one can, so it exercises absolute scale and the
foot-to-meter conversion.

### Sensitivity

Perturbing one exposure and re-measuring, on the real pilot data:

| perturbation | median residual |
|---|---|
| none (baseline) | 0.82 px |
| perspective center +0.25 m | 3.91 px |
| perspective center +1.0 m | 14.23 px |
| ω +0.01° | 6.12 px |
| ω +0.05° | 29.07 px |

The metric it replaces returns 0.00 px for every row of that table.

---

## 4. Results

### Frame coverage

Coverage is decided by projecting the eight building-volume corners onto each
sensor, not by asking whether a footprint polygon contains the AOI center point.
Restricted to the Fwd camera, two of five exposures fail: the building runs
249 px and 482 px off the sensor. Admitting the other camera heads brings the
usable set to **14 of 14 candidates**.

| camera | line | exposures |
|---|---|---|
| Bwd | 2034 | 4 |
| Color (nadir) | 2034 | 4 |
| Fwd | 2034 | 3 (of 5; 201822 and 201826 truncate) |
| Left | 2031 | 3 |
| Left | 2036 | 2 |
| Right | 2032 | 2 |

### View geometry

| | Fwd only | 4 oblique heads | + Color (nadir) |
|---|---|---|---|
| Exposures | 3 | 14 | **18** |
| Max view-ray separation | 8.9° | 103.3° | **103.3°** |
| Azimuth concentration R | 1.00 | 0.17 | **0.11** |
| Elevation range | 39–48° | 36–52° | **36–81°** |
| Near-nadir exposures | 0 | 0 | **3** |

R is the mean-resultant length of the viewing azimuths, weighted by
cos(elevation): 1.00 means every exposure looks from one bearing, and a nadir
view contributes nothing to it because its bearing is arbitrary. Azimuths fall
in clusters around 90°, 187°, 268° and 352° — all four facades are imaged —
and the Color head adds the top-down view.

### Reprojection

**Within group** (the camera-model check):

| group | exposures | median |
|---|---|---|
| Bwd, line 2034 | 4 | 0.90 px |
| Color, line 2034 | 4 | 0.73 px |
| Fwd, line 2034 | 3 | 0.97 px |
| Left, line 2031 | 3 | 1.14 px |
| Left, line 2036 | 2 | 1.69 px |
| Right, line 2032 | 2 | 0.92 px |
| **combined** | **18** | **1.02 px** (p90 2.58 px, 5,329 of 5,376 accepted, 99%) |

Every camera head is individually well modeled — including the Color head,
whose different focal length and sensor size are read straight from the EO
record. That is what confirms the principal-point sign fix.

**Across look directions** (reference `Color_2034_201831`, min NCC 0.7), by head:

| head | matches | median |
|---|---|---|
| Fwd | 873 | 0.87 px |
| Color | 857 | 1.87 px |
| Left | 662 | 3.28 px |
| Right | 254 | 4.03 px |
| Bwd | 429 | 4.07 px |
| **all anchors** | **3,075** | **2.10 px** (40% accepted) |
| **inside the building volume** | **1,631** | **2.96 px** (52% accepted) |

Color ties in better than the side heads: nadir-against-oblique is a smaller
view-angle change than oblique-against-opposing-oblique, so correlation is
easier.

**Read the second number.** Anchors span the whole AOI, roughly 140 m of parking
lot, road and treeline, and the flat surround matches *better* than the building
does — 1.6 px beyond 40 m against 11.7 px in the 20–30 m band at the roof's own
perimeter, where a planar patch straddles roof, parapet and wall. The
unrestricted median is therefore flattered by asphalt. See
[`WHAT_THE_QA_MEASURES.md`](WHAT_THE_QA_MEASURES.md).

### Triangulation vs LiDAR

| | Fwd only | 4 oblique heads | + Color |
|---|---|---|---|
| Points | 436 | 400 | 436 |
| Median 3D discrepancy | 0.70 m | 0.47 m | **0.26 m** |
| Median horizontal | 0.51 m | 0.36 m | **0.19 m** |
| Vertical bias | −0.29 m | +0.02 m | **−0.07 m** |
| Vertical sd | 1.52 m | 1.17 m | 1.06 m |
| Median views per point | 3 | 5 | 5.5 |
| Median max ray angle | 8.9° | 17.7° | **57.3°** |

The vertical bias collapsing from −0.29 m to −0.07 m shows the original figure
was weak-intersection noise, not a datum error. The nadir head more than triples
the ray intersection angle, which is what halves the 3D discrepancy again.

### Acceptance

| check | result | status |
|---|---|---|
| Usable exposures ≥ 3 | 18 frames | **PASS** |
| Within-group residual < 2.0 px | 1.02 px over 5,329 matches | **PASS** |
| Match acceptance rate ≥ 80% | 99% | **PASS** |
| View-ray separation ≥ 30° | 103.3° across 5 camera heads | **PASS** |
| Every crop contains the building | 18 / 18 | **PASS** |
| Crop padding on sensor ≥ 75% | min 0.786 (1 crop padded at the edge) | **PASS** |
| Source provenance with checksums | 3 assets, SHA-256 recorded | **PASS** |

### Why these numbers differ from the first published run

Every figure above was re-measured after three corrections, so they will not
match an earlier copy of this document:

* **The bounding volume's base is measured, not assumed.** It was
  `roof − 8 m`, which encodes a single-story building; it is now
  `ground_elevation_under`. The pilot's volume changes from 15.93 m to 13.06 m
  tall, and crops from 1600 px to 1648 px.
* **The LiDAR comes from Phase 2 COPC, not Phase 1 plain LAZ.** Tiles now
  resolve through `LidarTileIndex`, which prefers COPC. The AOI cloud grows from
  25,955 to 37,071 points, and gains seven years of currency.
* **The elevation pre-filter no longer clips.** `z_bounds_ft` defaulted to
  `[550, 750]`, the pilot's own range, which returned *zero* points for any AOI
  outside it.

The within-group residual, which is what tests the camera model, moved from
1.04 px to 1.02 px. The conclusion is unchanged.

## 5. Scope and limitations

- **Residuals are relative.** They measure agreement between exposures and with
  LiDAR. Absolute georeferencing accuracy is *not* established — that needs
  surveyed ground control, which this dataset does not include.
- **Cross-direction registration is 2.1 px overall and 3.0 px on the building,
  against 1.0 px within-group.** Phase 2 should expect
  to run a bundle adjustment across look directions rather than treating the
  delivered EO as final. The within-group numbers say the camera models are
  sound; the cross-direction numbers say the line-to-line ties are not tight.
- **One crop is padded at the sensor edge.** `Bwd_2034_201839` contains the
  whole building (8/8 corners) but its 1648 px window runs off the sensor, so
  21% of that crop is padding. The validity mask records it,
  and anything that uses the crop should honor the mask.
- **The published ortho is DTM-based, not a true ortho.** Measured with
  `scripts/check_ortho.py`: ground rectifies to 5.7 cm but roofs to 167 cm, a
  161 cm difference implying 16.1° off-nadir on this 5.61 m building. It
  therefore cannot be segmented for building footprints, and it carries no
  camera model, so it cannot serve as a reconstruction view either. Result in
  `docs/data/phase1_ortho_check.json` (written to `outputs/phase1/building_001/qa/ortho_check.json`).

## 6. What the crop package holds

`outputs/phase1/building_001/`

| file | contents |
|---|---|
| `images/frame_001..018.png` | 1648 × 1648 pinhole crops, five camera heads |
| `validity_masks/mask_001..018.png` | sensor-coverage masks |
| `cameras_normalized.json` | crop intrinsics, `R_cw`, `C_local`, coverage flags |
| `crop_manifest.json` | window offsets, required vs actual size |
| `coordinates.json` | local frame origin, CRS, datum |
| `check_features.json` | LiDAR-derived roof corners and building volume |
| `lidar_local.ply` | 37,071 points in the local metric frame |

QA evidence in `outputs/phase1/building_001/qa/`: `camera_findings.md`,
`reprojection_report.json`, `cross_view_matches.csv`, `triangulation_report.json`,
`cross_direction_report.json`, `view_geometry.json`, `frame_coverage.csv`,
`residual_field.png`, `residual_histogram.png`, `frame_coverage.png`,
`flight_geometry.png`, `building_in_crops.png`, `overlays/`.
