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

// start the COG adapter (a service worker) before VC Map asks for any tile
const index = resolve(dist, 'index.html');
const register = `    <script>
      if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js');
    </script>
  </head>`;
writeFileSync(index, readFileSync(index, 'utf8').replace('</head>', register));

const i = process.argv.indexOf('--tiles');
if (i > 0) {
  mkdirSync(resolve(dist, 'tiles'), { recursive: true });
  symlinkSync(resolve(process.argv[i + 1]), resolve(dist, 'tiles/downtown'), 'dir');
}
console.log(`site -> ${dist}`);
