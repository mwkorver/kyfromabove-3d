# Camera and imagery QA findings

- **Building**: Tractor Supply (Grayson, KY) (`building_001`)
- **Sensor family**: 434S82525X818326-f120_v2
- **Exposures used**: 18 of 18 covering the AOI
- **Look directions**: Bwd, Color, Fwd, Left, Right over flight lines 2031, 2032, 2034, 2036

## View geometry

- max view-ray separation: 103.3 deg
- elevation range: 36-81 deg
- azimuth concentration R = 0.11 (1.00 means every exposure looks from one bearing)

## Measured reprojection

Residuals are measured by rectifying image content onto each LiDAR anchor's
local surface plane in two exposures and locating the offset by normalised
cross-correlation. The imagery decides the result; no value is derived from
the camera model being tested.

**Within group** (same camera head and flight line) - the camera-model check:

- accepted matches: 5329 of 5376 (99%)
- median 1.02 px, p90 2.58 px, max 47.55 px
- median offset on the surface: 9.5 cm
- systematic component: 0.4 cm

**Across look directions** (reference Fwd_2034_201824) - also carries line-to-line georeferencing bias, and correlation is harder across wide view-angle changes, so this is reported rather than gated:

- accepted matches: 3075 of 7616 (40%)
- median 2.10 px, p90 51.15 px
- restricted to the building volume: 2.96 px over 1636 matches (52% accepted)

## Triangulation against LiDAR

- points triangulated: 436
- median 3D discrepancy: 0.26 m
- median horizontal: 0.19 m; vertical bias -0.07 m (sd 1.06 m)
- median max ray intersection angle: 57.3 deg

## Acceptance

| check | result | status |
|---|---|---|
| usable exposures >= 3 | 18 frames | **PASS** |
| within-group residual < 2.0 px | 1.02 px over 5329 matches | **PASS** |
| match acceptance rate >= 80% | 99% | **PASS** |
| view-ray separation >= 30 deg | 103.3 deg across 5 look directions | **PASS** |
| every crop contains the building | 18/18 | **PASS** |
| crop padding on sensor >= 75% | min 0.788; 1 crop(s) padded at the sensor edge, marked in the validity mask | **PASS** |

## Scope of these numbers

Residuals are relative: they measure agreement between exposures and with
LiDAR. Absolute georeferencing accuracy is not established, because the
dataset carries no surveyed ground control.
