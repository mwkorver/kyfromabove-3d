# Camera and imagery QA findings

- **Building**: Humana Building (Louisville, KY) (`humana`)
- **Sensor family**: 434S82525X818326-f120_v2
- **Exposures used**: 22 of 28 covering the AOI
- **Look directions**: Bwd, Color, Fwd, Right over flight lines 7019, 7020, 7021, 7022, 7023

## View geometry

- max view-ray separation: 97.5 deg
- elevation range: 39-76 deg
- azimuth concentration R = 0.20 (1.00 means every exposure looks from one bearing)

## Measured reprojection

Residuals are measured by rectifying image content onto each LiDAR anchor's
local surface plane in two exposures and locating the offset by normalised
cross-correlation. The imagery decides the result; no value is derived from
the camera model being tested.

**Within group** (same camera head and flight line) - the camera-model check:

- accepted matches: 5748 of 17108 (64%)
- median 3.19 px, p90 29.57 px, max 59.80 px
- median offset on the surface: 27.3 cm
- systematic component: 0.3 cm

**Across look directions** (reference Color_7021_46999) - also carries line-to-line georeferencing bias, and correlation is harder across wide view-angle changes, so this is reported rather than gated:

- accepted matches: 1812 of 27636 (17%)
- median 9.32 px, p90 60.93 px
- restricted to the building volume: 40.90 px over 25 matches (3% accepted)

## Triangulation against LiDAR

- points triangulated: 403
- median 3D discrepancy: 1.27 m
- median horizontal: 0.89 m; vertical bias 0.19 m (sd 2.70 m)
- median max ray intersection angle: 54.3 deg

## Acceptance

| check | result | status |
|---|---|---|
| usable exposures >= 3 | 22 frames | **PASS** |
| within-group residual < 2.0 px | 3.19 px over 5748 matches | **FAIL** |
| match acceptance rate >= 80% | 64% | **FAIL** |
| view-ray separation >= 30 deg | 97.5 deg across 4 look directions | **PASS** |
| every crop contains the building | 22/22 | **PASS** |
| crop padding on sensor >= 75% | min 0.646; 6 crop(s) padded at the sensor edge, marked in the validity mask | **FAIL** |

## Scope of these numbers

Residuals are relative: they measure agreement between exposures and with
LiDAR. Absolute georeferencing accuracy is not established, because the
dataset carries no surveyed ground control.
