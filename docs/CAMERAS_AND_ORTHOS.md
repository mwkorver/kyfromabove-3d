# Camera and ortho checks

The first phase of the project validated KyFromAbove's delivered camera models
against the imagery itself, and checked what kind of ortho each collection is.
These tools are still in the tree (`qa.py`, `ortho.py`, `scripts/validate_building.py`,
`scripts/check_ortho.py`); the phase reports are in [`history/`](history/).

## Using the library from Python

`Project()` finds the repository root itself, so nothing depends on the working
directory:

```python
from kyfromabove_3d import Project, build_planar_anchors, readers_for_cameras
from kyfromabove_3d.qa import measure_within_group_residuals

proj = Project()
anchors = build_planar_anchors(proj.lidar)
report = measure_within_group_residuals(anchors, proj.cameras,
                                        readers_for_cameras(proj.cameras))
print(report)   # median, p90 and max residual in pixels
```

## How reprojection accuracy is measured

Image content around each LiDAR anchor is resampled onto the anchor's local
surface plane in two exposures. Rectifying through a common plane removes the
perspective difference, so a correct camera model makes the two patches
identical; normalised cross-correlation measures the offset. The imagery decides
the result — no value is derived from the model under test.

The measure is sensitive: a 0.01° pointing error registers as 6.12 px.

## Checking an ortho before you trust it

Only a **true** ortho can be segmented directly for building footprints. A
standard, DTM-based ortho displaces every roof by `height * tan(theta)` — by an
amount that varies per building, jumps across mosaic seamlines, and cannot be
recovered from the ortho itself. It is rarely documented and invisible unless
measured.

```bash
python scripts/check_ortho.py
```

The check samples the same surface twice — through a validated exposure and
through the ortho's map projection — and correlates. **Ground anchors are the
control:** if they do not rectify to within a pixel, the measurement is
untrustworthy and no verdict is issued rather than a confident wrong one.

Results over the pilot building:

| collection | ground (control) | roof | verdict |
|---|---|---|---|
| Phase 3 `KY_KYAPED_2022_Season2_3IN` | 5.7 cm, scatter 8.0 | 167.1 cm | **standard ortho** |
| Phase 2 `KY_KYAPED_2019_6IN` | 280.7 cm, scatter 284.5 | 162.9 cm | **inconclusive** |

The 2022 collection is a standard, DTM-based ortho: +161.5 cm of roof
displacement, implying 16.1° off-nadir on a 5.61 m building. Not usable for
footprints.

The 2019 collection returns *inconclusive*, and the scatter says why: at 284.5 cm
it is as large as the 280.7 cm displacement, so the correlations are noise
rather than a shift. The cause is the three-year gap against a 2022 reference
exposure, not a defect in the ortho. Judging a 2019 product needs a 2019-era
reference frame. Note the check reports this as a scene mismatch rather than
blaming the georeferencing — a coherent shift and incoherent noise are different
failures and send you after different problems.

Use the raw nadir frames instead: same 7.6 cm GSD, a real camera model, and a
LiDAR roof plane to ray-cast onto.

## Pilot dataset

| | |
|---|---|
| Building | Tractor Supply Co., Grayson, Carter County, KY (`building_001`) |
| Footprint | 38.8 × 49.2 m, measured from 3,157 LiDAR roof returns |
| Flight lines | 2031, 2032, 2034, 2036, all flown 2022-11-21 |
| Cameras | Fwd, Bwd, Left, Right oblique heads plus the nadir Color head |
| Exposures | 18 usable, spanning 103° of view-ray separation |
| LiDAR | `N067E390.laz`, 25,955 points in AOI |

Residuals are relative — they measure agreement between exposures and with
LiDAR. Absolute georeferencing accuracy is not established; that needs surveyed
ground control, which this dataset does not include.
