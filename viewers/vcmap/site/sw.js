/* The COG adapter for VC Map's oblique view.
 *
 * VC Map asks for oblique tiles at obl/<collection>/<photo>/<z>/<x>/<y>.jpg on
 * its own grid: 512-pixel tiles, level 0 the coarsest (one tile = the whole
 * photo), x counted from the photo's left edge, y counted UP from its bottom
 * edge. KyFromAbove keeps each photo as a COG of 512-pixel JPEG tiles with
 * overviews at 1/2, 1/4 ... , counted DOWN from the top. This service worker
 * answers each VC Map tile from the COG straight away -- the matching 512-pixel
 * window of the matching overview, read with HTTP range requests (geotiff.js)
 * -- so the photos are never copied. A photo's height is rarely a whole number
 * of tiles, so a window can straddle two COG tile rows; geotiff.js reads both.
 * Outside the photo the tile is black (VC Map clips the layer to the photo).
 */
// geotiff.js's browser build names Worker as it loads (for its decoder pool,
// which this adapter does not use); a service worker has none.
self.Worker = self.Worker || class { constructor() { throw new Error('no workers in a service worker'); } };
importScripts('./geotiff.js');

const TILE = 512;
const ROUTE = /\/obl\/([^/]+)\/([^/]+)\/(\d+)\/(\d+)\/(\d+)\.jpg$/;
const lists = new Map(); // collection -> Promise<{photo: COG url}>
const photos = new Map(); // COG url -> Promise<{tiff, W, H, levels: [{w, h}]}> (most recent last)
const MAX_OPEN = 300;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const m = new URL(event.request.url).pathname.match(ROUTE);
  if (m) {
    event.respondWith(tile(event.request.url, m[1], decodeURIComponent(m[2]), +m[3], +m[4], +m[5]));
  }
});

function cogList(requestUrl, collection) {
  if (!lists.has(collection)) {
    const url = new URL(`../../../cogs.json`, requestUrl).href; // obl/<c>/cogs.json beside image.json
    lists.set(collection, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: ${r.status}`);
      return r.json();
    }));
  }
  return lists.get(collection);
}

function openPhoto(url) {
  if (photos.has(url)) {
    const p = photos.get(url);
    photos.delete(url); // keep the most recently used last
    photos.set(url, p);
    return p;
  }
  const p = (async () => {
    const tiff = await GeoTIFF.fromUrl(url, { cacheSize: 64 });
    const n = await tiff.getImageCount();
    const levels = [];
    for (let i = 0; i < n; i++) {
      const im = await tiff.getImage(i);
      levels.push({ w: im.getWidth(), h: im.getHeight() });
    }
    return { tiff, W: levels[0].w, H: levels[0].h, levels };
  })();
  p.catch(() => photos.delete(url));
  photos.set(url, p);
  if (photos.size > MAX_OPEN) photos.delete(photos.keys().next().value);
  return p;
}

async function tile(requestUrl, collection, name, z, x, y) {
  try {
    const cogs = await cogList(requestUrl, collection);
    const url = cogs[name];
    if (!url) return new Response('no such photo', { status: 404 });
    const { tiff, W, H, levels } = await openPhoto(url);
    // level z of VC Map's grid is overview (n - 1 - z): z = 0 the coarsest
    const k = Math.max(0, Math.min(levels.length - 1, levels.length - 1 - z));
    const r = 2 ** k; // full-resolution pixels per tile pixel at this level
    const { w, h } = levels[k];
    const sx = W / w, sy = H / h; // the overview's actual scale (r, give or take a rounding)
    // the tile in full-resolution pixels, rows counted down from the top
    const left = (x * TILE * r) / sx, right = ((x + 1) * TILE * r) / sx;
    const top = (H - (y + 1) * TILE * r) / sy, bottom = (H - y * TILE * r) / sy;
    const l = Math.max(0, Math.floor(left)), t = Math.max(0, Math.floor(top));
    const rr = Math.min(w, Math.ceil(right)), b = Math.min(h, Math.ceil(bottom));
    const canvas = new OffscreenCanvas(TILE, TILE);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, TILE, TILE);
    if (rr > l && b > t) {
      const image = await tiff.getImage(k);
      const rgb = await image.readRGB({ window: [l, t, rr, b], interleave: true });
      const cw = rr - l, ch = b - t;
      const rgba = new Uint8ClampedArray(cw * ch * 4);
      for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
        rgba[j] = rgb[i]; rgba[j + 1] = rgb[i + 1]; rgba[j + 2] = rgb[i + 2]; rgba[j + 3] = 255;
      }
      const src = new OffscreenCanvas(cw, ch);
      src.getContext('2d').putImageData(new ImageData(rgba, cw, ch), 0, 0);
      const fx = TILE / (right - left), fy = TILE / (bottom - top);
      ctx.drawImage(src, (l - left) * fx, (t - top) * fy, cw * fx, ch * fy);
    }
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.88 });
    return new Response(blob, { headers: { 'Content-Type': 'image/jpeg', 'Cache-Control': 'max-age=86400' } });
  } catch (err) {
    return new Response(String(err), { status: 502 });
  }
}
