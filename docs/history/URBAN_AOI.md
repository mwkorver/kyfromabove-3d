# building_002: a tower instead of a warehouse

*From the project's first phase (September 2026), kept as it was written,
with two corrections. Section 1 is wrong: the four oblique collections
together cover 99.6% of Kentucky, not only the cities
([the map](../figures/oblique_coverage.png), from each collection's own
outline). And section 7, "Nothing has been fetched or run", was written
before the run in section 6 and never updated.*

Everything here was measured by `scripts/find_urban_candidates.py`, which reads
the Phase 2 COPC cloud and the oblique frame index **without downloading
either**.

```bash
python scripts/find_urban_candidates.py --city Louisville
```

---

## 1. The constraint that decides everything

Kentucky's oblique imagery is **not statewide**. Four collections were flown
with barely-overlapping footprints:

| collection | reaches |
|---|---|
| 2022_Season2 | a 19 km-wide rural corridor — **no cities**. This is what `building_001` uses. |
| 2023_Season1 | Ashland, Richmond, Somerset, Pikeville |
| 2023_Season2 | Owensboro, Bowling Green |
| **2024_Season1** | **Louisville, Lexington, Covington, Frankfort, Paducah** |

The pilot is a Tractor Supply in Grayson because the collection we had contains
no city. Choosing a building before checking which collection flew it is how you
choose a building with no imagery.

## 2. Reading a 1.07 GB index without downloading it

`KY_KYAPED_2024_Season1_3IN_ImageFrames.gpkg` is 1.07 GB. It never has to come
down — a GeoPackage is SQLite with an RTree, so GDAL's `/vsicurl` serves a
spatial query by range request:

**81 frames over downtown Louisville in 4.2 seconds.** Same economics as the
COPC LiDAR.

### Two traps, both hit while building this

**A spatial `bbox` uses the RTree. An attribute `WHERE` does not.** A
`WHERE ID IN (...)` against the remote EO table scans the whole 553 MB file over
HTTP and does not finish. Query spatially, then filter the result in memory.

**The EO table is indexed on camera position, not on what the frame sees.** A
bbox drawn around the building returns **zero** EO records — the cameras are
~5,400 ft above and up to 7,700 ft away horizontally. Use a wide bbox (±9,000 ft
works here) and match against frame IDs afterwards.

**And the IDs carry the collection prefix.** The frames index gives
`KY_KYAPED_2024_Season1_3IN/Bwd_7021_47003.tif`; the EO `ID` is the same string
without `.tif`. Stripping the prefix to the bare shot id matches nothing —
silently, as an empty join rather than an error.

## 3. The target

**400 West Market (AEGON Center)** — `-85.75717, 38.25484`

LiDAR roof at 537 ft against 549 ft architectural, so the cloud lands where it
should. PNC Tower is the next cluster at 514 ft, two blocks away.

| | building_001 (pilot) | **building_002** |
|---|---|---|
| footprint | 71 × 74 m | **49 × 49 m** — *smaller* |
| height | 16 m | **165 m** — 10× |
| aspect | 4.5 : 1 wide | **1 : 3.4 tall** — inverted |
| roof as % of flying height | ~1% | **10%** |
| exposures over the AOI | 18 | **44** |
| sensor family | `434S82525X818326-f120_v2` | **identical** |

The sensor being the same matters: the Phase 1 camera model, interior
orientation and both calibration fixes carry over unchanged. This is a test of
*geometry and coverage*, not of a new instrument.

### Coverage is real but lopsided

| head | Bwd | Color | Fwd | Left | Right |
|---|---|---|---|---|---|
| frames | 10 | 10 | 9 | **4** | 11 |

Camera azimuth spread is 353° — effectively full surround — but **Left is thin
across every candidate in this downtown**, not just this one. Phase 1 got 103°
of ray separation from balanced heads; a lopsided set may not see every facade.
That is the first thing to check, and `find_urban_candidates.py` flags any head
under 5 frames for exactly this reason.

## 4. Why a tower is the harder test

Height bands of non-ground returns around the tower:

| band | share |
|---|---|
| 60–150 ft | **74.5%** |
| 150–250 ft | 8.0% |
| 250–350 ft | 3.2% |
| 350–450 ft | 4.9% |
| 450–600 ft | 9.4% |

Three-quarters of the cloud is the *surrounding low roofs*. The tower's own
facade range carries real but sparse returns.

That inverts the pilot. A 16 m warehouse is effectively all roof, and aerial
LiDAR measures a roof well. A tower is mostly facade by surface area, exactly
where the LiDAR is thinnest. **It isolates whether the oblique imagery is doing
genuine work.**

## 5. What is expected to break

Predictions recorded before running, so they can be wrong on the record.

1. **`required_extent` sizing.** The pilot needed 1611 × 1457 px for a 16 m
   volume. A 165 m volume at 1 : 3.4 aspect may not fit the source crop at all.
2. **Occlusion.** Neighboring towers block views; the isolated-building
   pipeline has never faced this. The validity masks model *sensor* coverage,
   not *occlusion*, so an occluded facade will look like valid pixels.
3. **The `Left` head.** Four frames may leave one facade unreconstructed.
4. **LiDAR on facades.** Sparse facade returns mean the wall planes have
   little to fit.

## 6. First validation run

`scripts/validate_building.py --config configs/louisville.yaml` runs end to end
against remote indexes. **Five of six gates pass; acceptance fails.**

```
[PASS] usable exposures >= 3: 18 frames
[PASS] within-group residual < 2.0 px: 1.89 px over 2841 matches
[FAIL] match acceptance rate >= 80%: 66%
[PASS] view-ray separation >= 30 deg: 99.1 deg across 4 look directions
[PASS] every crop contains the building: 18/18
[PASS] crop padding on sensor >= 75%: min 0.981
```

| | pilot | building_002 |
|---|---|---|
| within-group median | 1.04 px | **1.89 px** |
| cross-direction median | 1.54 px | **4.43 px** |
| triangulated vs LiDAR | 0.28 m | **1.82 m** |
| match acceptance | 99% | **66%** |
| look directions | 5 | **4** — `Left` never survives selection |

### Four bugs the pilot was hiding

Each returned a confident wrong answer rather than an error, because
`building_001` was the only AOI and its values had become defaults.

1. **`z_bounds_ft` defaulted to `[550, 750]`** — Grayson's elevation range.
   Louisville's ground is 463 ft, so the filter returned **zero points**.
2. **`Project.lidar` read a hardcoded `N067E390.laz`**, the Grayson tile.
3. **Collection and EO table name were hardcoded to 2022**, and the frames index
   was read whole rather than by bbox.
4. **`extract_roof_points(max_height_m=15.0)`** excludes a 165 m roof by
   construction, and `select="largest"` then returns whatever low structure
   shares the AOI. Raising the ceiling was not enough — clustering was also 2D,
   which merges a tower into the podium beneath it.

Before: a 43 × 96 m surface at 17 m. After: **164.1 m, 15.5 × 15.8 m**, pilot
unchanged. That work, and the separate discovery that the crown is a **dome**
so its "measured" rectangle corners sat off the surface entirely, is written up
in **[`ROOF_GEOMETRY.md`](ROOF_GEOMETRY.md)**.

### The acceptance failure is occlusion, and that is measured

Rejections are almost entirely `low_ncc` — 62.6% of cross-direction pairs. The
matches that *do* succeed are clean, at NCC 0.92 within-group. So the matcher is
rejecting genuinely non-corresponding patches rather than failing to align
corresponding ones.

Per-anchor acceptance is **bimodal**, which is the discriminating evidence:

| | |
|---|---|
| anchors never matched in any pair | **114 of 392 (29%)** |
| anchors always matched | 5 |
| median per-anchor acceptance | 29% |

A wrong camera model degrades every anchor roughly equally. A third of anchors
being unmatchable **from every direction** is a visibility property of the site,
not an error in the geometry.

Acceptance by look-direction pair says the same thing:

| within-group | | cross-direction | |
|---|---|---|---|
| Bwd–Bwd | 81% | Color–Right | 26% |
| Fwd–Fwd | 79% | Bwd–Color | 28% |
| Right–Right | 59% | Color–Fwd | 30% |
| Color–Color | 57% | Color–Color | 42% |

Frames from the same head and line see nearly the same scene; opposed heads do
not, because towers stand between them.

**The 80% gate encodes an assumption that every anchor is visible from every
view.** That holds for an isolated rural warehouse and is simply false downtown.
The threshold has deliberately **not** been changed — lowering a gate until it
passes is how a pipeline stops measuring anything. What the number needs is
either an occlusion-aware acceptance model, or a separate urban threshold
justified by measurement.

### Still open

- The volume is the tower's **crown only** — 245 m² against roughly 2,400 m² for
  its main mass, because AEGON Center steps inward near the top. See
  [`ROOF_GEOMETRY.md`](ROOF_GEOMETRY.md) §6.
- `Fwd` and `Bwd` within-group residuals are 5.58 and 4.18 px against Color's
  0.97. The anchors are independent planar patches, not roof-derived, so this
  survived the roof fix unchanged. Consistent with the roof sitting at 10% of
  flying height rather than ~1%, but not yet demonstrated.

## 7. Status

**The AOI is defined and its inputs are confirmed available. Nothing has been
fetched or run.**

- `configs/louisville.yaml` loads and resolves `building_002`
- 44 exposures confirmed present, all five heads, EO matched 44/44
- Phase 2 COPC covers it — 4 tiles, statewide, unchanged from the pilot

Next is `scripts/validate_building.py` against this config, which needs the
imagery: 44 frames at ~45 MB each is ~2 GB, or far less using COG windowed
reads, which is what `imagery.py` already does.
