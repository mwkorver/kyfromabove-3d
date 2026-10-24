# Finding a roof, and describing it honestly

The roof drives three things: the bounding volume, the crop sizing that follows
from it, and the check features used to validate reprojection. Get the roof
wrong and everything downstream is confidently wrong, because none of these
steps error — they return a plausible answer about the wrong surface.

Two separate problems, each with its own failure mode:

1. **Finding the right roof** when an AOI holds more than one.
2. **Describing it** without asserting a shape nobody measured.

Both were invisible until `building_002`, because the pilot AOI holds exactly
one flat rectangular roof, and every shortcut is correct for that case.

---

## 1. Finding the right roof

### Three silent filters

`extract_roof_points` isolates roofs geometrically — the KyFromAbove point
cloud publishes no ASPRS class 6, so there is nothing to select on (see
[`LIDAR_SOURCES.md`](LIDAR_SOURCES.md) §5). Three of its parameters can each
discard the building you want, without saying so:

| parameter | default | what it silently excludes |
|---|---|---|
| `max_height_m` | 15.0 | **any roof above 15 m.** A 165 m tower has no roof at all under this ceiling. |
| `min_normal_z` | 0.90 | curved or steep surfaces. On the tower this keeps **466 of 4,475** points in the top 15 m — only the gentle apex cap survives. |
| `max_planarity_rms_m` | 0.15 | locally rough surfaces: mechanical plant, parapets, crowns. |

`max_height_m` is now `Optional`; pass `None` for no ceiling, or set
`aoi.roof.max_height_m` in the config to the AOI's real vertical extent.

### `largest` is the wrong rule downtown

`roof_check_features` keeps one connected cluster. Which one is a choice:

* **`largest`** — most points. Correct for an isolated structure, and what the
  pilot uses.
* **`tallest`** — highest median elevation. Correct in a dense core, where a
  tower's roof is small and its podium and neighbors are large, so `largest`
  reliably picks the wrong building.

Set with `aoi.roof.select`.

### Connectivity must be 3D

Raising the ceiling and asking for `tallest` **still** returned a 21.4 m roof
for a 165 m building. The clustering used `cKDTree(roof_xyz[:, :2])` — two
dimensions.

In plan view a tower's roof sits *inside* its podium's footprint. A 2D graph
therefore merges surfaces 144 m apart into one cluster, and that cluster's
median elevation belongs to whichever surface has more points: always the
lower, larger one. `tallest` then faithfully returns the podium.

Clustering on `roof_xyz[:, :3]` separates them. For an isolated single-story
building nothing is stacked, so this changes nothing.

| | selected roof |
|---|---|
| 2D connectivity | 43 × 96 m at **17 m** — the podium |
| 3D connectivity | 15.5 × 15.8 m at **164.1 m** — the tower |
| pilot, either way | 40.5 × 51.7 m at **1.31 m** — unchanged |

---

## 2. Describing it honestly

### 400 West Market is a dome

Radial profile of the tower's crown, from the point cloud:

| radius | mean z | | radius | mean z |
|---|---|---|---|---|
| 0–2 m | 155.7 m | | 10–13 m | 151.9 m |
| 2–4 m | 154.8 m | | 13–16 m | 148.8 m |
| 4–6 m | 157.4 m | | 16–20 m | 142.2 m |
| 6–8 m | 153.8 m | | 20–26 m | 138.9 m |

The selected cluster spans **4.46 m vertically over an 8.3 m radius** and falls
**2.71 m from center to edge**. It is not a plane.

### What a rectangle does to a dome

`cv2.minAreaRect` produced four corners, all placed at the median elevation
**164.06 m**, recorded as `provenance="lidar:minAreaRect"` — the project's label
for a measured position.

They were not measured. A rectangle around a circular footprint puts its
corners at **radius × √2**, so those four points sat *outside the dome
entirely*. Not three meters too high — over no roof surface at all, in a
project whose whole discipline is refusing to assert what it has not measured.

### The planarity test

`roof_check_features` now fits a plane before assuming one:

| roof | plane-fit RMS |
|---|---|
| pilot, flat warehouse | **0.192 m** |
| 400 West Market crown | **1.496 m** |
| threshold (`max_plane_rms_m`) | 0.5 m |

An 8× separation, so the default sits well clear of both.

**Below threshold** — the rectangle path, unchanged: four corners plus a
centroid at the median elevation. Honest, because the surface is flat there.

**Above threshold** — surface-sampled features. Each is snapped to the **local**
median elevation of nearby returns, and any sample position **without enough
support is not emitted at all**:

```
RoofApex        (-0.68,  0.56, 164.82)   lidar:apex
RoofSurface_N   ( 0.06,  6.29, 161.59)   lidar:surface_sample
RoofSurface_S   ( 0.06, -6.29, 164.48)   lidar:surface_sample
RoofSurface_W   (-6.23, -0.00, 164.57)   lidar:surface_sample
```

Four, not five. **`RoofSurface_E` is absent** for lack of measured support, and
refusing to place it is the behavior that matters. The 2.9 m spread between N
and S is the dome's real asymmetry, which one median elevation erased.

---

## 3. Extent and check features are different questions

The first attempt at the above was wrong, and the error is worth keeping.

Replacing the rectangle corners with surface samples was right for check
features — but the same four points were also driving the **footprint**. The
volume collapsed from 15.5 × 15.8 m to 5.8 × 12.6 m, and `ground_elevation_under`
crashed looking for feature ids starting with `RoofCorner`.

They answer different questions:

| | question | source |
|---|---|---|
| **extent** (`roof_footprint`) | how much ground does this cover? | `minAreaRect` of the roof cloud — honest for any roof, flat or domed |
| **check features** (`roof_check_features`) | where exactly is the surface? | measured points on it |

A minimum-area rectangle is a perfectly good bounding box for a dome. It is
only dishonest as a set of surface positions. Conflating the two is precisely
how four points that touch nothing came to be labeled as measurements.

`roof_footprint` supplies the extent. The pilot's footprint is **bit-identical**
to what the code produced before the split — max corner difference **0.0 m** —
because for a flat roof both paths run the same `minAreaRect` on the same points.

---

## 4. The invariant, and how it is tested

> **A check feature is a measurement or it is nothing.**

`tests/test_roof_selection.py` pins this on a synthetic podium-plus-dome scene:

* every emitted feature sits within 0.5 m of measured surface
* a dome never receives `minAreaRect` corners, and never reports one elevation
  for every feature
* the rectangle corners **demonstrably** land off the dome — all four are shown
  to fall beyond the footprint with no surface beneath them, rather than the
  test merely asserting that they would
* a flat roof still gets rectangle corners, and `largest` and `tallest` agree
  when nothing is stacked
* 3D connectivity separates the stacked podium and tower

The dome fixture needed tuning to be realistic: at a 4.5 m rise the
`min_normal_z` filter rejects the whole surface, which is exactly why the real
crown survives only as its apex cap.

---

## 5. Configuration

```yaml
aoi:
  roof:
    max_height_m: 180.0     # null for no ceiling; 15.0 suits a single storey
    select: tallest         # or "largest" for an isolated building
```

Defaults are `15.0` and `largest`, so `building_001` is unaffected.

## 6. Still open

`select: tallest` finds AEGON Center's **crown**, not its main mass. The
building steps inward near the top, so the volume is 27.7 × 28.0 × 16.5 m built
on a 245 m² footprint, against roughly 2,400 m² for the tower proper. Crops are
sized for the crown.

That is defensible — it is a real surface, measured — but it under-uses the
imagery. The honest fix is a union of roof levels rather than a single cluster,
which needs a way to decide which stacked surfaces belong to one building. That
is the same question [`URBAN_AOI.md`](URBAN_AOI.md) reaches from the other
direction, and the reason Overture GERS footprints are on the roadmap.
