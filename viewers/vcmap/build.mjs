// The VC Map test site as plain static files in dist/: VC Map's prebuilt app
// (@vcmap/ui's dist), our settings and the COG adapter (site/), geotiff.js for
// the adapter, and a line in index.html that starts the adapter. The oblique
// collection goes in afterwards with scripts/oblique_collection.py --out
// dist/obl/<collection>.
//
//   node build.mjs                                   # the site alone
//   node build.mjs --tiles ../../outputs/downtown    # and local 3D Tiles at dist/tiles/downtown (a link)
import { cpSync, rmSync, readFileSync, writeFileSync, mkdirSync, symlinkSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, 'dist');
rmSync(dist, { recursive: true, force: true });
cpSync(resolve(here, 'node_modules/@vcmap/ui/dist'), dist, { recursive: true });
cpSync(resolve(here, 'site'), dist, { recursive: true });
cpSync(resolve(here, 'node_modules/geotiff/dist-browser/geotiff.js'), resolve(dist, 'geotiff.js'));

// start the COG adapter (a service worker) before VC Map asks for any tile, and
// show any tile it could not make in a line at the bottom of the page; our
// page scripts are named by their contents so a browser never runs a stale copy
const index = resolve(dist, 'index.html');
const scripts = ['oblique-fill.js', 'wireframe.js'].map((name) => {
  const version = createHash('sha1').update(readFileSync(resolve(dist, name))).digest('hex').slice(0, 10);
  return `    <script src="./${name}?v=${version}"></script>`;
}).join('\n');
const register = `    <script>
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('./sw.js');
        navigator.serviceWorker.addEventListener('message', (event) => {
          const message = event.data && event.data.kyfaTileError;
          if (!message) return;
          let line = document.getElementById('kyfa-tile-error');
          if (!line) {
            line = document.createElement('div');
            line.id = 'kyfa-tile-error';
            line.style.cssText = 'position:fixed;left:8px;bottom:28px;z-index:9999;max-width:70vw;padding:4px 8px;'
              + 'font:12px sans-serif;background:#fff3f3;color:#a00;border:1px solid #d99;border-radius:4px;';
            line.onclick = () => line.remove();
            document.body.appendChild(line);
          }
          line.textContent = 'Photo tile failed: ' + message + ' (click to hide)';
        });
      }
    </script>
${scripts}
  </head>`;
writeFileSync(index, readFileSync(index, 'utf8').replace('</head>', register));

// the places the Cesium viewer flies to (configs/places.json, one list for both
// viewers): a viewpoint each, 800 m out as there, listed under "Places" in the
// layer list, where a click flies the map to it
{
  const file = resolve(dist, 'config/kyfromabove.config.json');
  const cfg = JSON.parse(readFileSync(file, 'utf8'));
  const { places } = JSON.parse(readFileSync(resolve(here, '../../configs/places.json'), 'utf8'));
  const slug = (name) => name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const start = cfg.viewpoints.find((v) => v.name === cfg.startingViewpointName);
  cfg.viewpoints = [start, ...places.map((p) => ({
    name: `place-${slug(p.name)}`, type: 'Viewpoint', groundPosition: [p.lon, p.lat, p.ground_m],
    distance: 800, heading: 15, pitch: -35, roll: 0, animate: true, duration: 2,
    properties: { title: p.name },
  }))];
  const at = cfg.contentTree.findIndex((c) => c.name === 'imagery');
  cfg.contentTree.splice(at, 0, { name: 'places', type: 'NodeContentTreeItem', title: 'Places', initOpen: true },
    ...places.map((p) => ({
      name: `places.${slug(p.name)}`, type: 'ViewpointContentTreeItem', viewpointName: `place-${slug(p.name)}`,
      title: p.name,
    })));
  writeFileSync(file, JSON.stringify(cfg, null, 2));
}

const i = process.argv.indexOf('--tiles');
if (i > 0) {
  mkdirSync(resolve(dist, 'tiles'), { recursive: true });
  symlinkSync(resolve(process.argv[i + 1]), resolve(dist, 'tiles/downtown'), 'dir');
}

// --cloud: the page served beside the viewer on CloudFront (/viewer/vcmap/):
// the published statewide trees instead of local tiles -- every product, one
// shown at a time -- and the published terrain under the 3D map and the
// oblique collection (ground heights)
const PRODUCTS = [
  ['overture', 'Footprints, Overture height'],
  ['boxes', 'Footprints, LiDAR height'],
  ['measured', 'Measured (2)'],
  ['complete', 'Completed (3)'],
  ['complete_tint', 'Completed, added in pink'],
  ['neural', 'Borrowed fill (4)'],
  ['neural_tint', 'Borrowed fill, added in violet'],
];
if (process.argv.includes('--cloud')) {
  const file = resolve(dist, 'config/kyfromabove.config.json');
  const cfg = JSON.parse(readFileSync(file, 'utf8'));
  const measured = cfg.layers.find((l) => l.name === 'measured');
  const at = cfg.layers.indexOf(measured);
  cfg.layers.splice(at, 1, ...PRODUCTS.map(([name, title]) => ({
    ...measured, name, url: `/kentucky/${name}/tileset.json`, screenSpaceError: 16,
    activeOnStartup: name === 'measured', properties: { title },
  })));
  const item = cfg.contentTree.findIndex((c) => c.name === 'buildings.measured');
  cfg.contentTree.splice(item, 1,
    ...PRODUCTS.map(([name]) => ({ name: `buildings.${name}`, type: 'LayerContentTreeItem', layerName: name })));
  cfg.layers.push({
    type: 'TerrainLayer', name: 'terrain', url: '/terrain/', activeOnStartup: true,
    requestVertexNormals: false, properties: { title: 'LiDAR ground' },
  });
  cfg.obliqueCollections.forEach((c) => c.dataSets.forEach((d) => { d.terrainProvider = { url: '/terrain/' }; }));
  writeFileSync(file, JSON.stringify(cfg, null, 2));
}
console.log(`site -> ${dist}`);
