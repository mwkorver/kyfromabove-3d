# The Lambda image for the pipeline's steps (tile, piece, building).
# Build context: the repository root (deploy/stack.py sets it).

# basisu, the KTX2 texture encoder the meshes need, built for ARM Linux at
# the version the laptop runs (Homebrew's 2.50); nothing else of this stage
# reaches the image.
FROM public.ecr.aws/lambda/python:3.12 AS basisu
RUN dnf install -y cmake gcc-c++ make git && dnf clean all
RUN git clone --depth 1 --branch v2_50 https://github.com/BinomialLLC/basis_universal.git /src/basisu \
 && cmake -S /src/basisu -B /src/basisu/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /src/basisu/build -j"$(nproc)" --target basisu \
 && find /src/basisu -type f -name basisu -perm -u+x -exec cp {} /usr/local/bin/basisu \;

# Python 3.12's image is Amazon Linux 2023: 3.11's is Amazon Linux 2, whose
# glibc 2.26 is too old for today's numpy wheels (pip falls back to compiling
# numpy, and there is no compiler).
FROM public.ecr.aws/lambda/python:3.12

# The pipeline's libraries, pinned to the laptop's versions
# (deploy/requirements-lambda.txt); OpenCV without its display libraries,
# which Lambda's Linux lacks. geopandas, rasterio and pyproj bring their own
# GDAL and PROJ in their wheels.
COPY deploy/requirements-lambda.txt /tmp/requirements-lambda.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lambda.txt

# rasterio's GDAL needs libexpat, which the minimal Lambda image leaves out.
RUN dnf install -y expat && dnf clean all

# The package itself, without its declared dependencies (they would pull in
# the desktop OpenCV).
COPY pyproject.toml README.md ${LAMBDA_TASK_ROOT}/pkg/
COPY src ${LAMBDA_TASK_ROOT}/pkg/src
RUN pip install --no-cache-dir --no-deps ${LAMBDA_TASK_ROOT}/pkg && rm -rf ${LAMBDA_TASK_ROOT}/pkg

# The Phase 2 LiDAR tile grid (22 MB): which COPC file holds which box.
COPY data/raw/kyfromabove_phase2_pointcloud_5k_grid.gpkg /opt/data/raw/
# and Phase 3's (4 MB): a tile takes the phase closest in time to its obliques
COPY data/raw/kyfromabove_phase3_pointcloud_5k_grid.gpkg /opt/data/raw/

# The two grids that place a building: NAVD88 heights to the ellipsoid
# (GEOID99 over Kentucky) and NAD83 to NAD83(HARN). They go beside pyproj's
# own proj.db and the network stays off: without them PROJ does not fail, it
# quietly places the building 1.2 m or 34 m wrong (export_3dtiles.to_ecef
# stops instead). The laptop fetches the same two files into its cache.
RUN d="$(python -c 'import pyproj; print(pyproj.datadir.get_data_dir())')" \
 && for g in us_noaa_g1999u07.tif us_noaa_nadcon5_nad83_1986_nad83_harn_conus.tif; do \
      curl -fsSL -o "$d/$g" "https://cdn.proj.org/$g" || exit 1; done
ENV PROJ_NETWORK=OFF

COPY --from=basisu /usr/local/bin/basisu /usr/local/bin/basisu

# The building step runs the laptop's scripts, each in its own process.
COPY scripts ${LAMBDA_TASK_ROOT}/scripts
ENV SCRIPTS_DIR=${LAMBDA_TASK_ROOT}/scripts
# and the classes' settings (configs/class_strategies.yaml): without them every
# building would be built as before classes (kyfromabove_3d.settings now stops)
COPY configs ${LAMBDA_TASK_ROOT}/configs
ENV KFA_CONFIG_DIR=${LAMBDA_TASK_ROOT}/configs

COPY deploy/handlers.py ${LAMBDA_TASK_ROOT}/

# Lambda's home is read-only: matplotlib's and PROJ's caches go to /tmp.
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PROJ_USER_WRITABLE_DIRECTORY=/tmp/proj
# GDAL's block cache defaults to 5% of the RAM it can see, which in a 3 GB
# Lambda may be the host's; fixed at 256 MB it cannot crowd a large
# building's crops. Speed only: the pixels are the same.
ENV GDAL_CACHEMAX=256
CMD ["handlers.make_pieces"]
