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

// Finished tiles are kept in Cache Storage, so a place already looked at loads
// at once on a later visit. Change the name when the drawing below changes; the
// old store is then dropped. Kept to about MAX_TILES tiles (~60 KB each), the
// oldest dropped first.
const TILE_CACHE = 'kyfa-obl-tiles-v1';
const MAX_TILES = 3000;
let added = 0;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil((async () => {
  for (const name of await caches.keys()) {
    if (name.startsWith('kyfa-obl-tiles-') && name !== TILE_CACHE) await caches.delete(name);
  }
  await self.clients.claim();
})()));

self.addEventListener('fetch', (event) => {
  const m = new URL(event.request.url).pathname.match(ROUTE);
  if (m) {
    event.respondWith(tile(event.request.url, m[1], decodeURIComponent(m[2]), +m[3], +m[4], +m[5]));
  }
});

// The photo list (obl/<c>/cogs.json beside image.json) is kept in Cache Storage
// once read: the browser stops an idle service worker and starts it afresh, and
// the fresh one then needs no request to the (password-protected) site. A failed
// read is not kept, so the next tile tries again.
const LIST_CACHE = 'kyfa-obl-lists-v1';

async function readList(url) {
  const cache = await caches.open(LIST_CACHE);
  const kept = await cache.match(url);
  if (kept) return kept.json();
  const r = await fetch(url, { credentials: 'include' });
  if (!r.ok) throw new Error(`photo list ${url}: HTTP ${r.status}`);
  await cache.put(url, r.clone());
  return r.json();
}

function cogList(requestUrl, collection) {
  if (!lists.has(collection)) {
    const url = new URL(`../../../cogs.json`, requestUrl).href;
    const p = readList(url);
    p.catch(() => lists.delete(collection));
    lists.set(collection, p);
  }
  return lists.get(collection);
}

// failures go to the open pages, which show them (see build.mjs)
async function report(message) {
  for (const client of await self.clients.matchAll()) client.postMessage({ kyfaTileError: message });
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
  const cache = await caches.open(TILE_CACHE);
  const kept = await cache.match(requestUrl);
  if (kept) return kept;
  const made = await makeTile(requestUrl, collection, name, z, x, y);
  if (made.ok) {
    await cache.put(requestUrl, made.clone());
    if (++added % 100 === 0) await trim(cache);
  }
  return made;
}

async function trim(cache) {
  const keys = await cache.keys(); // oldest first
  for (const key of keys.slice(0, Math.max(0, keys.length - MAX_TILES))) await cache.delete(key);
}

async function makeTile(requestUrl, collection, name, z, x, y) {
  let cogs;
  try {
    cogs = await cogList(requestUrl, collection);
  } catch (err) {
    await report(String(err.message || err));
    return new Response(String(err), { status: 503 });
  }
  const url = cogs[name];
  if (!url) return new Response('no such photo', { status: 404 });
  try {
    return await drawTile(url, z, x, y);
  } catch {
    photos.delete(url); // open the photo afresh and try once more
    try {
      return await drawTile(url, z, x, y);
    } catch (err) {
      await report(`${name} level ${z} tile ${x},${y}: ${err.message || err}`);
      return new Response(String(err), { status: 502 });
    }
  }
}

async function drawTile(url, z, x, y) {
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
}
