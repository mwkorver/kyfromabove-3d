# Camera and imagery QA findings

- **Building**: PNC Tower (Louisville, KY) (`pnc_tower`)
- **Sensor family**: 434S82525X818326-f120_v2
- **Exposures used**: 19 of 26 covering the AOI
- **Look directions**: Bwd, Color, Fwd, Right over flight lines 7019, 7020, 7021, 7022, 7023

## View geometry

- max view-ray separation: 96.7 deg
- elevation range: 39-78 deg
- azimuth concentration R = 0.25 (1.00 means every exposure looks from one bearing)

## Measured reprojection

Residuals are measured by rectifying image content onto each LiDAR anchor's
local surface plane in two exposures and locating the offset by normalised
cross-correlation. The imagery decides the result; no value is derived from
the camera model being tested.

**Within group** (same camera head and flight line) - the camera-model check:

- accepted matches: 4187 of 9515 (72%)
- median 1.59 px, p90 25.17 px, max 64.61 px
- median offset on the surface: 13.8 cm
- systematic component: 0.7 cm

**Across look directions** (reference Color_7021_46999) - also carries line-to-line georeferencing bias, and correlation is harder across wide view-angle changes, so this is reported rather than gated:

- accepted matches: 2554 of 15570 (33%)
- median 3.47 px, p90 38.09 px
- restricted to the building volume: 2.63 px over 1065 matches (39% accepted)

## Triangulation against LiDAR

- points triangulated: 239
- median 3D discrepancy: 0.91 m
- median horizontal: 0.74 m; vertical bias 0.05 m (sd 2.36 m)
- median max ray intersection angle: 64.0 deg

## Acceptance

| check | result | status |
|---|---|---|
| usable exposures >= 3 | 19 frames | **PASS** |
| within-group residual < 2.0 px | 1.59 px over 4187 matches | **PASS** |
| match acceptance rate >= 80% | 72% | **FAIL** |
| view-ray separation >= 30 deg | 96.7 deg across 4 look directions | **PASS** |
| every crop contains the building | 19/19 | **PASS** |
| crop padding on sensor >= 75% | min 0.804; 4 crop(s) padded at the sensor edge, marked in the validity mask | **PASS** |

## Scope of these numbers

Residuals are relative: they measure agreement between exposures and with
LiDAR. Absolute georeferencing accuracy is not established, because the
dataset carries no surveyed ground control.
