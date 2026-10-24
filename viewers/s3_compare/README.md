# Standalone S3 Compare Viewer

A 100% client-side, serverless 3D comparison viewer designed for static hosting on **Amazon S3** (or CloudFront / GitHub Pages).

It runs completely without a Python backend server.

---

## Key Features

1. **Dual-Pane Synchronized 3D Comparison**:
   - Left: 3D textured mesh, its shape from the LiDAR and its colour from the aerial frames.
   - Right: Overture footprint boxes (with dynamic toggle between LiDAR-derived heights and Overture heights).
2. **In-Browser Real-Time S3 Oblique Slicing (`geotiff.js`)**:
   - When a building is clicked, the viewer uses `geotiff.js` to perform **HTTP range requests directly against Kentucky's public S3 bucket** (`https://kyfromabove.s3.amazonaws.com`).
   - Only the specific sub-window pixels for the building are downloaded and decoded on the fly into an HTML `<canvas>`.
   - **Zero server and zero crop storage cost** on your hosting bucket!
3. **Layer Options**:
   - OpenStreetMap base map (the standard picture tiles).
   - **USGS National Map NAIP Imagery WMS** (`USGSNAIPImagery:NaturalColor`).
4. **Interactive North Arrow**:
   - Real-time needle rotation tracking camera heading.
   - Click to smoothly animate camera to North-up (`heading: 0`).
5. **Static Ground Elevation (DEM)**:
   - Drapes all tiles on a 100% static heightmap terrain derived from the LiDAR DEM (`data/terrain/`).

**The data is not in the repository.** `data/` (frames, crops, terrain, building
index) and `tilesets/` are written on your own machine by
`scripts/export_s3_demo.py` (below); only the page, this README and the test
server are checked in.

---

## High-Performance Oblique Image Architecture

The viewer supports a **hybrid two-tier architecture** for oblique images:

1. **Instant Static JPEG Tier (~5 ms)**:
   - Pre-cut JPEGs located in `data/crops/<gers>/<index>_1600.jpg` and `_320.jpg`.
   - When present, the browser loads these directly via static HTTP GET with zero decoding latency.
   - Written by `export_s3_demo.py --precut`; the author's last export had crops for 51 downtown buildings.

2. **Cloud-Optimized GeoTIFF (COG) Live Slicing Tier (~50–150 ms)**:
   - When static crops are absent, the viewer slices live from the 2024 oblique photos in KyFromAbove's public bucket (stored as Cloud-Optimized GeoTIFFs), using `geotiff.js`.
   - **TIFF Header Caching**: Header metadata is parsed exactly once per aerial exposure.
   - **Internal Overview Pyramid Matching**: Kentucky obliques have 5 downsampled overview levels (factors 2, 4, 8, 16, 32).
     - **Thumbnails** (`maxPx = 200`): Select overview factor 16 or 32, reducing tile reads from 49 tiles down to **1 single tile (~20 KB)**!
     - **Main Frame** (`maxPx = 1600`): Selects factor 1 or 2, reducing tile reads by up to 75%.
   - **Connection Priority & Cancellation Queue**:
     - The selected high-res main frame is loaded first, preventing HTTP connection starvation.
     - Filmstrip thumbnails are loaded sequentially in the background.
     - Clicking a different building or closing the panel cancels any pending background thumbnail tasks immediately.

---

## Static Asset Export Script

To re-export or update metadata, terrain, or pre-cut crops:

```bash
# Fast metadata update (skipping terrain tile re-export):
.venv/bin/python scripts/export_s3_demo.py --skip-terrain

# Pre-cut and cache all building crops offline before uploading:
.venv/bin/python scripts/export_s3_demo.py --precut --skip-terrain

# Copy crop files instead of symlinks (for S3 upload preparation):
.venv/bin/python scripts/export_s3_demo.py --copy-crops --skip-terrain
```

---

## Local Testing

To test the demo locally before uploading to S3:

```bash
# Start the static test server
python viewers/s3_compare/test_server.py --port 8790
```

Then open your browser to:
[http://127.0.0.1:8790/](http://127.0.0.1:8790/)

---

## Uploading to Amazon S3

1. **Ensure your target S3 bucket is configured for Static Website Hosting** (or served via CloudFront).
2. **Sync the directory to S3**:
   Use the `--follow-symlinks` flag so that the 3D tileset and crop assets are copied into the bucket:

```bash
aws s3 sync viewers/s3_compare/ s3://YOUR-BUCKET-NAME/ \
    --follow-symlinks \
    --exclude "test_server.py" \
    --exclude "*.DS_Store"
```

3. Open the static website endpoint:
   `http://YOUR-BUCKET-NAME.s3-website-YOUR-REGION.amazonaws.com/`

