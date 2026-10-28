# What the reprojection QA actually measures

*From the project's first phase (September 2026), kept as it was written.
The checks it describes are still in the tree
([CAMERAS_AND_ORTHOS.md](../CAMERAS_AND_ORTHOS.md)); the meshes are checked
by the texture check and the audit ([AUDIT.md](../AUDIT.md)).*

Phase 1's headline numbers — within-group residual, cross-direction residual,
3D discrepancy — are all computed on **planar anchors** selected from the LiDAR.
Which surfaces become anchors therefore decides what those numbers describe, and
that turned out not to be the building.

---

## 1. Restricting to the building volume

Anchors are selected across the whole AOI, which for the pilot is roughly 140 m
of parking lot, road and treeline. `qa.anchors_in_volume` restricts a report to
anchors inside the bounding volume, and both numbers are now printed:

```
[qa] cross-direction, all anchors : median 2.10 px over 3075 matches (40% accepted)
[qa] cross-direction, in volume   : median 2.96 px over 1631 matches (52% accepted)
```

The measurement is not re-run. `ReprojectionReport.filter_anchors` re-summarizes
the same matches over a smaller population, so the two numbers are exactly
comparable.

### The pilot got worse, not better

The prediction was that restricting to the building would *lower* the pilot's
2.10 px. It went **up, to 2.96 px**.

The reasoning behind the prediction was wrong in a specific way worth recording.
The anchor-visibility map colors each anchor by **acceptance rate**, and the
pilot's roof is green while its surround is red — which was read as "the roof
matches well". Acceptance rate is not residual. The residual-versus-distance
figure, built in the same batch, says the opposite:

| distance from center | median residual |
|---|---|
| 0–10 m | 2.8 px |
| 10–20 m | 3.9 px |
| **20–30 m** | **11.7 px** |
| 30–40 m | 1.9 px |
| 40–80 m | 1.6 px |

**The building region is the worst part of the frame; the flat surround is the
best.** Anchors far from the building match *less often*, but the ones that do
match are on clean asphalt and land within a couple of pixels. So the
unrestricted median was being flattered by parking lot.

2.96 px is the more honest figure. The metric was too generous before, not too
harsh now.

## 2. The tower's QA was not measuring the tower

```
[qa] cross-direction, in volume : median 63.60 px over 3 matches (1% accepted)
```

Three matches. Of `building_002`'s **392 anchors, only 24 fall inside the tower
volume**, all between 140 and 143.6 m, and all labeled `roof`.

Not one anchor sits on a facade. So the tower's published Phase 1 numbers —
1.89 px within-group, 4.43 px cross-direction, 1.82 m 3D discrepancy — were
measured on **surrounding low roofs and ground**, not on the building.

That reframes the 66% acceptance failure. Occlusion is real and the visibility
map shows it clustering directionally, but the measurement it degrades was
never describing the tower in the first place.

### Why: two structural assumptions

`build_planar_anchors` encodes a horizontal building.

1. **`min_normal_z = 0.80`** — anchors must be within ~37° of horizontal. A
   vertical facade has `normal_z ≈ 0` and is rejected by construction.
2. **The grid is planimetric.** Cells are formed on `(x, y)` with `per_cell=2`,
   so every facade point up a 165 m tower lands in the same cell. Even with
   normals allowed, a whole column would yield two anchors.

Neither is a tuning problem. They are the shape of the assumption.

## 3. Facade anchors are feasible, and the margin is large

Measured on the tower's shaft — `20 ≤ r < 30 m`, `40 ≤ z < 120 m`, which is the
main mass below the crown's inward step:

| | |
|---|---|
| points on the shaft | **8,459** |
| near-vertical (`\|n_z\| < 0.35`) | 7,896 |
| also planar at RMS ≤ 0.20 m | **6,002** |
| median local-plane RMS | **0.104 m** |
| distinct 8 m³ cells | **260** → up to 520 anchors at `per_cell=2` |

Against **24** anchors from the current method. The facade is better sampled
than the crown, and flatter than the roof filter's own 0.08 m threshold would
suggest is available.

Neighborhood size matters: `k=8` gives median RMS 0.104 m, `k=16` gives 0.168,
`k=32` gives 0.211. A facade is locally flat but globally curved around the
building, so a tight neighborhood fits it and a wide one straddles corners.

> A sampling correction worth keeping: an initial pass queried `10 < r < 18 m`
> and found **nothing** below 105 m, which read as "the facades are not sampled".
> That shell was *inside* the building — the tower steps inward near the top, so
> its crown is at r < 20 m while its shaft is at r 20–30 m. The facades are
> sampled throughout, at roughly 2,000 returns per 20 m band.

### What implementing them requires

* a near-vertical eligibility branch alongside the horizontal one, with its own
  planarity threshold (~0.20 m, against 0.08 for roofs)
* **3D cell keys** — `(x, y, z)` rather than `(x, y)` — so anchors spread up a
  facade instead of collapsing into one column
* `k=8` for the vertical branch
* a `facade` label beside `roof` and `ground`, so reports can separate them
* verification that plane-rectified NCC behaves on vertical patches: the tangent
  basis is normal-agnostic, but nothing has tested it at grazing incidence, and
  oblique views see a facade far more obliquely than they see a roof

The last point is the real unknown and should be measured before the rest is
built on top of it.

## 4. Facade anchors, built

`build_planar_anchors(include_facades=True)` adds the vertical pass. Default is
off, so `building_001` is untouched.

On the tower it produces **612 anchors on the shaft, spanning 35–141 m** —
roughly 100 per 20 m band — against 24 before, all of which were on the crown.

### Do they match? Yes, once the test is posed correctly

The first measurement used reference `Color_7021_46999` and returned 5%
acceptance at 26 px. That test was invalid: `Color` is the **nadir** head, and a
nadir camera sees a vertical wall edge-on. Every pair was doomed by geometry,
not by the method.

Against oblique references:

| reference | facade | roof control |
|---|---|---|
| Color (nadir) | 5% · 26.19 px | 24% · 3.33 px |
| **Right** | 6% · **3.81 px** | 11% · 6.41 px |
| Fwd | 10% · 16.89 px | 34% · 3.64 px |
| Bwd | 7% · 27.47 px | 40% · 7.13 px |

Against `Right` the facade matches at **3.81 px — better than the roof control
in the same run**. So plane-rectified NCC does work at grazing incidence.

### Acceptance is capped at 30% by geometry, not by matching

A wall faces one way. Measured across the 612 shaft anchors and 19 cameras,
**only 30% of pairs have the camera on the visible side** — median 5 cameras per
anchor, and every anchor is seen by at least 2, so triangulation is possible.

Read against that ceiling, the observed 6–10% becomes **20–33%**, which is the
roof control's range. **The matcher needs back-face culling for facade
anchors**: without it, 70% of attempts are geometrically impossible and are
counted as `low_ncc` rejections, which reads as the method failing.

### Orienting the normal took three attempts

The sign of a facade normal is what says which side is visible, and
`fit_local_planes` returns an arbitrary one. Two plausible rules are wrong:

* **Local neighborhood centroid** — sits *in* the wall plane, because on a
  hollow cloud the neighborhood is the wall. 68% correct on the tower.
* **Occupancy probe** — "LiDAR never sees inside a building, so the emptier side
  is outside". False: the outside of a wall is air and is equally empty. **60%**
  on a clean hollow box, barely above chance.
* **Cross-section centroid** — points in a thin z-slab within 40 m. A closed
  cross-section has its centroid inside it. Over 90% on the box, **88% on the
  tower shaft**, where the remaining errors are on setback faces and
  neighboring structures caught in the shaft band.

### The two blockers are cleared

Both were fixed in `210d016`. The volume is now built from the building's whole
surface rather than its crown, so the shaft's anchors fall inside it, and the
matcher culls pairs where the camera is behind the wall instead of counting them
as `low_ncc` rejections.

With both in place, on `building_002` (`outputs/urban/building_002/qa/reprojection_report.json`,
within-group, 1,433 anchors over 18 frames):

| surface | matches | median error |
|---|---|---|
| **facade** | **2,433** | **1.70 px** |
| ground | 1,621 | 2.43 px |
| roof | 1,220 | 1.55 px |

Overall acceptance **69.6%**, against 33% before back-face pairs were excluded
from the denominator rather than counted as failures.

**The facade is the largest and best-matched surface on the tower** — twice the
roof's matches at the same accuracy. That is the opposite of where this section
started, which was 24 anchors, none of them on the building.

Cross-direction is a harder test and the facade does worse there: 132 matches at
17.5 px, against the roof's 998 at 4.26 px. A wall is seen well by the heads
that face it and not at all by the others, so cross-*head* agreement on a facade
is thin by geometry. Within-group is the number that says whether the surface
reconstructs.

## 5. Status

`anchors_in_volume` and `filter_anchors` are in place and both numbers are
reported. **The in-volume figure is reported, not gated on.** That began as a
workaround for the tower having three in-volume matches; with the volume and the
culling fixed it has 5,274 accepted matches, so the reason to leave it ungated
is now weaker than it was.

What the facade evidence settled, for the project rather than the QA: **facades
stay in the deliverable.** They are the biggest surface on a tall building, they
match as well as roofs, and the LiDAR has no color on them any more than it
does anywhere else. Building selection carries a facade term alongside the roof
complexity score — see `docs/SURVEY_ZONES.md`. It also rules out a 2.5D TIN as
the reconstruction target, since a 2.5D surface cannot represent a vertical
face.
