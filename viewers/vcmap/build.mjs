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
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, 'dist');
rmSync(dist, { recursive: true, force: true });
cpSync(resolve(here, 'node_modules/@vcmap/ui/dist'), dist, { recursive: true });
cpSync(resolve(here, 'site'), dist, { recursive: true });
cpSync(resolve(here, 'node_modules/geotiff/dist-browser/geotiff.js'), resolve(dist, 'geotiff.js'));

// start the COG adapter (a service worker) before VC Map asks for any tile, and
// show any tile it could not make in a line at the bottom of the page
const index = resolve(dist, 'index.html');
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
  </head>`;
writeFileSync(index, readFileSync(index, 'utf8').replace('</head>', register));

const i = process.argv.indexOf('--tiles');
if (i > 0) {
  mkdirSync(resolve(dist, 'tiles'), { recursive: true });
  symlinkSync(resolve(process.argv[i + 1]), resolve(dist, 'tiles/downtown'), 'dir');
}

// --cloud: the page served beside the viewer on CloudFront (/viewer/vcmap/):
// the published statewide tree instead of local tiles, and the published
// terrain under the 3D map and the oblique collection (ground heights)
if (process.argv.includes('--cloud')) {
  const file = resolve(dist, 'config/kyfromabove.config.json');
  const cfg = JSON.parse(readFileSync(file, 'utf8'));
  const measured = cfg.layers.find((l) => l.name === 'measured');
  Object.assign(measured, { url: '/kentucky/measured/tileset.json', screenSpaceError: 16 });
  measured.properties.title = 'Measured (product 2)';
  cfg.layers.push({
    type: 'TerrainLayer', name: 'terrain', url: '/terrain/', activeOnStartup: true,
    requestVertexNormals: false, properties: { title: 'LiDAR ground' },
  });
  cfg.obliqueCollections.forEach((c) => c.dataSets.forEach((d) => { d.terrainProvider = { url: '/terrain/' }; }));
  writeFileSync(file, JSON.stringify(cfg, null, 2));
}
console.log(`site -> ${dist}`);
