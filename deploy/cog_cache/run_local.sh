#!/bin/sh
# The frame cache on this machine, in a container: sh deploy/cog_cache/run_local.sh [slice] [cache dir]
# GDAL then reads through it with COG_CACHE_URL=http://127.0.0.1:8001
SLICE=${1:-1m}
DIR=${2:-$(pwd)/outputs/cog_cache}
mkdir -p "$DIR/log"   # the cache itself on a Docker volume: a folder shared from macOS is slow for small reads
docker rm -f kfa-cog-cache >/dev/null 2>&1
docker run -d --name kfa-cog-cache -p 127.0.0.1:8001:8001 \
  -e SLICE="$SLICE" -e CACHE_MAX=20g \
  -v "${TEMPLATE:-$(cd "$(dirname "$0")" && pwd)/nginx.conf.template}:/etc/nginx/templates/cog.conf.template:ro" \
  -v kfa-cog-cache:/var/cache/nginx/cog -v "$DIR/log:/var/log/nginx" \
  nginx:1.27-alpine >/dev/null && echo "frame cache up: slice $SLICE, $DIR"
