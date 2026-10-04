# Camera and imagery QA findings

- **Building**: 400 West Market / AEGON Center (Louisville, KY) (`building_002`)
- **Sensor family**: 434S82525X818326-f120_v2
- **Exposures used**: 18 of 19 covering the AOI
- **Look directions**: Bwd, Color, Fwd, Right over flight lines 7019, 7020, 7021, 7022, 7023

## View geometry

- max view-ray separation: 99.1 deg
- elevation range: 39-81 deg
- azimuth concentration R = 0.15 (1.00 means every exposure looks from one bearing)

## Measured reprojection

Residuals are measured by rectifying image content onto each LiDAR anchor's
local surface plane in two exposures and locating the offset by normalised
cross-correlation. The imagery decides the result; no value is derived from
the camera model being tested.

**Within group** (same camera head and flight line) - the camera-model check:

- accepted matches: 5274 of 15763 (70%)
- median 1.75 px, p90 25.35 px, max 59.25 px
- median offset on the surface: 15.3 cm
- systematic component: 0.4 cm

**Across look directions** (reference Color_7021_46999) - also carries line-to-line georeferencing bias, and correlation is harder across wide view-angle changes, so this is reported rather than gated:

- accepted matches: 2271 of 24361 (29%)
- median 4.74 px, p90 50.18 px
- restricted to the building volume: 33.23 px over 271 matches (9% accepted)

## Triangulation against LiDAR

- points triangulated: 507
- median 3D discrepancy: 1.22 m
- median horizontal: 0.75 m; vertical bias -0.12 m (sd 4.77 m)
- median max ray intersection angle: 10.0 deg

## Acceptance

| check | result | status |
|---|---|---|
| usable exposures >= 3 | 18 frames | **PASS** |
| within-group residual < 2.0 px | 1.75 px over 5274 matches | **PASS** |
| match acceptance rate >= 80% | 70% | **FAIL** |
| view-ray separation >= 30 deg | 99.1 deg across 4 look directions | **PASS** |
| every crop contains the building | 18/18 | **PASS** |
| crop padding on sensor >= 75% | min 0.908; 2 crop(s) padded at the sensor edge, marked in the validity mask | **PASS** |

## Scope of these numbers

Residuals are relative: they measure agreement between exposures and with
LiDAR. Absolute georeferencing accuracy is not established, because the
dataset carries no surveyed ground control.
