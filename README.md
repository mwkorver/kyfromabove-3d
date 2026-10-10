# kyfromabove-3d

[![CI](https://github.com/mwkorver/kyfromabove-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/mwkorver/kyfromabove-3d/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!NOTE]
> **Public prototype, released under [MIT](LICENSE): use, fork, and adapt it
> freely.** It is provided as-is, with no support or active maintenance; issues
> and pull requests are not monitored, so please fork rather than wait on
> changes here. The pilot covers four areas around Louisville, not all of
> Kentucky.

**Every building in Kentucky in 3D, from the state's own open data.** The
shape of each building comes from KyFromAbove's LiDAR, its colour from
KyFromAbove's oblique aerial photographs, and its outline and identity from
Overture's building footprints. The result is published as 3D Tiles and
viewed in a web browser.

![Downtown Louisville: this project's buildings on the left, Overture's boxes on the right](docs/figures/compare_downtown_from_north.jpg)

*Downtown Louisville from the north. Left: this project, walls and roofs
from the LiDAR, coloured from the aerial frames. Right: the same footprints
as plain boxes at Overture's heights, what most 3D maps show.*

[KyFromAbove](https://kyfromabove.ky.gov) is the Commonwealth of Kentucky's
aerial imagery and LiDAR programme, published as open data on AWS. This is
an independent project built *on* that data, not a product of the programme;
nothing here is endorsed by it.

## What it makes

The same buildings, four ways. The viewer has a button for each, so you can
switch between them in place and see what each step adds:

| View | Shape | Colour |
|---|---|---|
| **Overture** | Overture's footprint, pushed up to Overture's own height | coloured by where that height came from: Overture's figure, floors × 3.5 m, or a 6 m guess |
| **LiDAR heights** (product 1) | the same footprint, pushed up to the height the LiDAR measured | plain grey |
| **Measured** (product 2) | walls and roof from the LiDAR | from the aerial frames, only where a frame actually saw the surface |
| **Completed** (product 3) | the same as Measured | what no frame saw is filled in, from the same building's own frames only, and labelled as filled |

Nothing is invented from outside the building's own KyFromAbove data: no
internet photos, no generic textures. Clicking a building says which LiDAR
survey and which photo collection it was made from, how many months apart
they were taken, and where its height came from. More in
[`docs/PRODUCTS.md`](docs/PRODUCTS.md).

## How it works

![The pipeline: stage 0 and four steps, each writing to one store](docs/figures/pipeline.png)

The unit of work is one LiDAR tile, a square about 1.5 km across; Kentucky
has 46,167 of them, 41,566 with buildings on them. Every step reads from and
writes to one *store*: a bucket in AWS for real runs, or a folder on a laptop
for testing. The same code runs in both; the tile step was checked to give
byte-identical results in each.

| Step | What it does | Runs on | Code |
|---|---|---|---|
| 0 | Groups Overture's footprints by the LiDAR tile they fall in | a laptop, once per Overture release | `scripts/stage0_index.py` |
| 1 · tile | Measures every building's height from the LiDAR, splits outlines that hold two or three buildings, and picks the LiDAR survey and photo collection taken **closest together in time** | one Lambda per tile | `pieces.tile_step`, `sources_choice.py` |
| 2 · piece | Cuts each building's LiDAR and lists the aerial frames that see it | Lambdas, 25 buildings each | `pieces.piece_step` |
| 3 · building | Fits walls to the LiDAR, colours them from the frames, builds the measured and completed meshes, and audits them | EC2 Spot machines, 16 buildings at a time | `building.py`, `scripts/build_mesh.py` |
| 4 · publish | Batches each tile's buildings into 3D Tiles, builds the statewide tree and the terrain, writes the viewer | a Spot machine, or a laptop | `publish.py`, `terrain.py` |

CloudFront serves the published files (behind a password while this is a
demo) to a CesiumJS page in [`viewer/`](viewer/). The cloud setup, the costs
measured along the way, and everything that went wrong are in
[`docs/CLOUD.md`](docs/CLOUD.md).

## Where it stands (September 2026)

A pilot ran every step end to end in AWS:

- **9,252 buildings built as textured meshes** in four areas: downtown
  Louisville, the St. Matthews suburb, Hazard in the eastern hills, and a
  rural edge near Somerset.
- **315,970 buildings as boxes** across the 255 LiDAR tiles around
  Louisville, so the view from far away has real content.
- **Terrain** under all of it, and the viewer with its four views.
- **Total AWS cost: $6.73.**

What it showed:

- Buildings sit on the ground: at 10 buildings across the four areas, the
  typical gap between a building's base and the terrain is under 0.2 m.
- The page draws at 60 frames a second.
- The first version downloaded far too much (up to 190 MB for one view),
  almost all of it grey boxes. The box files are now 13 to 32 times smaller.
  The textured files are the next to slim down.

For all of Kentucky (2.79 million Overture buildings), the building step,
the expensive one, is estimated at about $430 on Spot machines. Overture
lists about 5.3 million buildings in a rectangle drawn around Kentucky, but
that rectangle takes in parts of the neighbouring states; the 2.79 million
are the ones on Kentucky's own LiDAR tiles.

## Status and limits

- A prototype: it ran end to end for four areas around Louisville, not for the
  whole state. The statewide figures above are estimates, not a run.
- The live demo is behind a password and is not public. To see it, run the
  [one-tile example](#one-tile-on-a-laptop) below, or read the figures here.
- Built and tested on one laptop (macOS) and one AWS account. Nothing here has
  been tried on Windows.
- No support, and no promise to merge changes; see
  [CONTRIBUTING.md](CONTRIBUTING.md).

## Why the LiDAR gives the shape and the photos only the colour

KyFromAbove's Phase 2 LiDAR covers the whole state at about one return every
0.6 m on a roof, and it has no colour. The oblique photos are much finer (3
inches a pixel, 25 times finer than the LiDAR on a wall the scanner reached)
but a photo is a poor source of *shape*:

| Building | Widest angle between views of a point | Error of points triangulated from the photos (median, against the LiDAR) |
|---|---:|---:|
| Tractor Supply, Grayson (low, corrugated wall) | 57° | **0.26 m** |
| PNC Tower | 64° | 0.91 m |
| Humana Building | 54° | 1.27 m |
| 400 West Market | 10° | 1.22 m |

The angle is there; the matching fails. A curtain wall repeats the same
window every 1.2 to 1.5 m, fifty floors high, so a match that slips one bay is
a metre wrong with a confident score, and glass reflects a different sky in
every view.

So depth is never taken from a picture. Each wall is fitted as a plane to the
LiDAR's own returns on it, and colour is read from the best-facing frame that
can actually see each point, with neighbours tested for blocking the view
([`docs/BUILDING_MESH.md`](docs/BUILDING_MESH.md)).

## What Overture is for

Footprints, and the GERS id that every file in the project is keyed on. Two
things about Overture's data in Kentucky were found the hard way:

- **Its heights are reliable only above about 40 m.** Below that, only about
  one height in seven is within 20% of the LiDAR. Every height used here comes
  from the LiDAR; the Overture view shows the difference.
- **`roof_shape` is almost empty**, under 0.5% of buildings at every size.

More in [`docs/SURVEY_ZONES.md`](docs/SURVEY_ZONES.md) and
[`docs/CASE_COMPLEXITY.md`](docs/CASE_COMPLEXITY.md).

## Running it

### Set up

```bash
uv venv .venv --python python3.11 && source .venv/bin/activate
uv pip install -e ".[overture,cloud]" && uv pip install pytest
```

The LiDAR tile indexes come from KyFromAbove's public bucket; no AWS account
is needed to read it:

```bash
mkdir -p data/raw && for p in 2 3; do aws s3 cp --no-sign-request s3://kyfromabove/elevation/PointCloud/TileGrids/kyfromabove_phase${p}_pointcloud_5k_grid.gpkg data/raw/; done
```

The two geoid grids that place buildings at the right height are fetched by
PROJ on first use. Without them, PROJ silently places a building 1.2 m or
34 m wrong, so the code stops instead.

### Test

```bash
pytest -q
```

About 320 tests, a minute, no network needed.

### One tile on a laptop

A corner of downtown Louisville (72 of the buildings in tile N076E229),
building five of them as meshes and the rest as boxes. Stage 0 fetches the
footprints from Overture's release on S3 (about half a minute); the rest took
about two minutes on a laptop:

```bash
python scripts/stage0_index.py --bbox -85.764 38.25 -85.75 38.26 --store outputs/one_tile --tiles N076E229
python scripts/tile_step.py --store outputs/one_tile --tiles N076E229
python scripts/piece_step.py --store outputs/one_tile --tiles N076E229
python scripts/building_step.py --inputs outputs/one_tile --store outputs/one_tile --tiles N076E229 --sample 5 --workers 5
python scripts/publish_step.py --inputs outputs/one_tile --tiles N076E229 --tree --terrain --viewer --start -85.7522 38.25 700
python -m http.server -d outputs/one_tile 8000
```

Then open <http://localhost:8000/viewer/index.html>. The LiDAR and the
frames are read straight from KyFromAbove's public bucket.

### In AWS

The stacks are in [`deploy/`](deploy/) (AWS CDK): the tile, piece and building
steps, the Spot machines, and the CloudFront viewer. The runners
(`deploy/run_tile_step.py`, `run_piece_step.py`, `run_spot_build.py`) start a
step over a list of tiles and tally what it costs as it goes; `run_spot_build.py`
stops the machines before a spending cap. [`docs/CLOUD.md`](docs/CLOUD.md)
walks through a full run.

Before deploying your own copy: the bucket name `kyfromabove-3d` in the code
and docs is the author's and bucket names are global, so change it; the Spot
runner's cap is a check in the script, so deploy the budget stack too
(`docs/CLOUD.md`, "Watching the spend"); and a deployed viewer is public unless the password is switched on. See
[SECURITY.md](SECURITY.md).

### Checking a change

A change to the meshing or colouring code is checked against 19 buildings
that have broken before, without looking at them:

```bash
python scripts/audit_reference.py rebuild && python scripts/audit_reference.py compare
```

It fails, naming the building and the measure, if anything got worse
([`docs/AUDIT.md`](docs/AUDIT.md)).

## Documents

| Document | Covers |
|---|---|
| [`docs/CLOUD.md`](docs/CLOUD.md) | The pipeline in AWS, step by step: what each step costs, the pilot, the bugs it found, serving the views |
| [`docs/PRODUCTS.md`](docs/PRODUCTS.md) | The three products: what is measured, what is inferred, what is invented and how it is labelled |
| [`docs/BUILDING_MESH.md`](docs/BUILDING_MESH.md) | How a building becomes a textured mesh: wall bands, curved walls, the roof, textures |
| [`docs/AUDIT.md`](docs/AUDIT.md) | Checking a code change against the buildings that have broken before |
| [`docs/DOWNTOWN_RUN.md`](docs/DOWNTOWN_RUN.md) | Every footprint of downtown Louisville, built as a batch on a laptop |
| [`docs/LIDAR_SOURCES.md`](docs/LIDAR_SOURCES.md) | KyFromAbove's three LiDAR phases, their density and dates, and two silent traps |
| [`docs/SURVEY_ZONES.md`](docs/SURVEY_ZONES.md) | Which buildings have roofs worth densifying and walls worth imaging, statewide |
| [`docs/CASE_COMPLEXITY.md`](docs/CASE_COMPLEXITY.md) | The three Louisville towers measured on one scale |
| [`docs/ROOF_GEOMETRY.md`](docs/ROOF_GEOMETRY.md) | Finding a roof when an area holds several, and describing a domed crown honestly |
| [`docs/URBAN_AOI.md`](docs/URBAN_AOI.md) | Where the oblique imagery is, and how the first tower was chosen |
| [`docs/WHAT_THE_QA_MEASURES.md`](docs/WHAT_THE_QA_MEASURES.md) | What the camera QA actually measures, and where it was fooled |
| [`docs/CAMERAS_AND_ORTHOS.md`](docs/CAMERAS_AND_ORTHOS.md) | Validating the delivered camera models, and checking what kind of ortho a collection is |
| [`docs/history/`](docs/history/) | The first phase's report, as written |

## Layout

```text
src/kyfromabove_3d/
  pieces.py            steps 1 and 2: a LiDAR tile into per-building pieces
  product1.py          product 1's measures: every building's height from the LiDAR
  sources_choice.py    which LiDAR survey and photo collection a tile uses
  building.py          step 3: a piece in, a building's meshes out
  facade_fill.py       the completed product: fill what no frame saw, from the building itself
  publish.py           step 4: a tile's buildings as batched 3D Tiles, and the statewide tree
  terrain.py           the ground under the buildings, as the viewer's terrain
  store.py             where a step's results go: a folder, or S3
  audit.py, texture_check.py   a built mesh against evidence it was not built from
  lidar.py, imagery.py, footprints.py, coordinates.py, cameras.py, crops.py, prepare.py, ...
scripts/
  stage0_index.py, tile_step.py, piece_step.py, building_step.py, publish_step.py   the steps, locally
  build_mesh.py, colorize_building.py, check_building.py   one building's mesh, colour, checks
  overture_boxes.py    Overture footprints: fetched, measured on the LiDAR, split, boxed
  audit_reference.py   the reference-set audit
  intake_server.py     local viewers: /tiles, /compare, frames and crops, terrain
  render_flyaround.py  a synthetic orbit or walk round a built building, with its true camera path
  score_trajectory.py  camera poses estimated from video against the true ones: ATE, RPE, scale, drift
  video_poses.py       a video model's (LingBot-Map's) camera for every frame, run on a Mac or a GPU
  register_video.py    a real video's true camera path, from landmarks and tracking against the meshes
  wall_visibility.py   which aerial frames see each slice of wall, clearly and face-on enough to use
  ...                  surveys, diagnostics, the downtown batch run
deploy/                the AWS stacks (CDK), the Lambda handlers, the runners, cost models
viewer/index.html      the published viewer (CesiumJS, four views)
tools/                 local viewer pages: tiles, compare, point cloud
configs/               the landmark buildings, the audit's reference set, per-building configs
tests/                 no network needed
docs/                  the documents above, and their figures
```

## Licence

The code is MIT; see [LICENSE](LICENSE). The data is licensed separately, and
the two sources differ:

- **KyFromAbove** imagery and LiDAR: public domain.
- **Overture** building footprints: the buildings theme is primarily under the
  [Open Database Licence (ODbL)](https://docs.overturemaps.org/attribution/),
  which includes a share-alike condition, and it draws on several sources that
  each ask for credit. The tiles and tilesets this project publishes contain
  Overture's footprints, so anything you publish from them is likely to carry
  the same terms. Check them before you reuse the output; this README is not
  legal advice.
- **OpenStreetMap** base map in the viewer: © OpenStreetMap contributors.

## Data source and acknowledgement

All imagery and elevation data here comes from
**[KyFromAbove](https://kyfromabove.ky.gov/)**, the Commonwealth of Kentucky's
aerial photography and elevation program, via the `kyfromabove` S3 bucket in
the [AWS Registry of Open Data](https://registry.opendata.aws/kyfromabove/):
the Phase 2 and Phase 3 LiDAR (COPC), the oblique and nadir aerial frames of
the 2022-2024 collections (read as COGs), and their exterior orientation and
frame outline tables. Building footprints come from
[Overture Maps](https://overturemaps.org/) (the buildings theme, under its
published licence terms). The viewer's base map is ©
[OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.

The KyFromAbove data is **public domain**: no licence, no attribution
requirement. It is credited here anyway, and named in this repository,
because what makes the project possible is unusual: very few U.S. states
publish oblique aerial frames openly, and fewer still publish the exterior
orientation tables that make them photogrammetrically usable. Kentucky
publishes both, alongside statewide LiDAR, at no cost to anyone. In 2025 it
became the
[first state in the nation to complete statewide oblique aerial coverage](https://www.kentucky.gov/Pages/Activity-stream.aspx?n=FinanceandAdministrationCabinet&prId=266).

Errors in this repository are ours, not the data provider's.
