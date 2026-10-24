/* A "Wireframe" switch for the buildings in the 3D view: draws every triangle's
 * edges instead of the textured surface (Cesium's debugWireframe; the layers ask
 * Cesium to keep the edges, see enableDebugWireframe in the config). Shown only
 * while the 3D map is the one in view; ?wireframe=1 starts with it on.
 */
(function wireframeSwitch() {
  let on = new URLSearchParams(location.search).get('wireframe') === '1';

  function watch(app) {
    const button = document.createElement('button');
    button.id = 'kyfa-wireframe';
    button.style.cssText = 'position:fixed;right:64px;bottom:36px;z-index:9999;padding:4px 10px;cursor:pointer;'
      + 'font:12px sans-serif;border-radius:4px;border:1px solid #888;';
    const show = () => {
      button.textContent = on ? 'Wireframe: on' : 'Wireframe: off';
      button.style.background = on ? '#2a78d6' : '#fff';
      button.style.color = on ? '#fff' : '#222';
      button.style.display = app.maps.activeMap?.className === 'CesiumMap' ? 'block' : 'none';
    };
    // every tileset of every 3D Tiles layer, whichever is showing
    const apply = () => {
      for (const layer of app.layers) {
        if (layer.className !== 'CesiumTilesetLayer') continue;
        for (const map of app.maps) {
          for (const impl of layer.getImplementationsForMap(map)) {
            if (impl.cesium3DTileset) impl.cesium3DTileset.debugWireframe = on;
          }
        }
      }
    };
    button.onclick = () => { on = !on; apply(); show(); };
    document.body.appendChild(button);
    // a layer turned on makes its tileset when it loads, after this event
    app.layers.stateChanged.addEventListener(() => setTimeout(apply, 0));
    app.maps.mapActivated.addEventListener(() => { apply(); show(); });
    apply();
    show();
  }

  // the VC Map app is made after this page's scripts run
  const seen = new Set();
  const timer = setInterval(() => {
    for (const app of window.vcs?.apps?.values() ?? []) {
      if (!seen.has(app)) {
        seen.add(app);
        watch(app);
      }
    }
    if (seen.size) clearInterval(timer);
  }, 250);
})();
