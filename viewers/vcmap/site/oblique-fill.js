/* The oblique view shows one photo at a time and nothing around it, so zooming
 * out past the photo leaves it floating in white. Two limits keep the photo
 * filling the window:
 *  - the furthest-out zoom is the one at which the photo just covers the window
 *    (worked out again for every photo and whenever the window changes size);
 *  - on a zoom, the view slides inward as far as needed to keep the window
 *    inside the photo. Panning is left free, so dragging towards an edge still
 *    reaches the next photo (VC Map switches to the photo whose centre is
 *    nearest the middle of the view, see switchThreshold in the config).
 */
(function fillWindowWithPhoto() {
  function hook(map) {
    const ol = map.olMap;
    const views = new WeakSet();

    const limitZoom = () => {
      const image = map.currentImage;
      const size = ol.getSize();
      if (!image || !size || !size[0] || !size[1]) return;
      const view = ol.getView();
      const [w, h] = image.meta.size;
      const min = view.getZoomForResolution(Math.min(w / size[0], h / size[1]));
      view.setMinZoom(min);
      if (view.getZoom() < min) view.setZoom(min);
    };

    const keepInside = () => {
      const image = map.currentImage;
      const size = ol.getSize();
      const view = ol.getView();
      const center = view.getCenter();
      const res = view.getResolution();
      if (!image || !size || !center || !res) return;
      const [w, h] = image.meta.size;
      const hx = Math.min((size[0] * res) / 2, w / 2);
      const hy = Math.min((size[1] * res) / 2, h / 2);
      const x = Math.min(Math.max(center[0], hx), w - hx);
      const y = Math.min(Math.max(center[1], hy), h - hy);
      if (x !== center[0] || y !== center[1]) view.setCenter([x, y]);
    };

    const onView = () => {
      const view = ol.getView();
      if (!views.has(view)) {
        views.add(view);
        // after the zoom has also moved the centre (a zoom about the mouse sets
        // the resolution first, then the centre, in one go)
        view.on('change:resolution', () => {
          queueMicrotask(() => {
            if (ol.getView() === view) keepInside();
          });
        });
      }
      limitZoom();
    };

    ol.on('change:view', onView);
    ol.on('change:size', limitZoom);
    map.imageChanged?.addEventListener(limitZoom);
    onView();
  }

  function watch(app) {
    const hooked = new Set();
    const onMap = (map) => {
      if (map.className !== 'ObliqueMap' || hooked.has(map) || !map.olMap) return;
      hooked.add(map);
      hook(map);
    };
    app.maps.mapActivated.addEventListener(onMap);
    if (app.maps.activeMap) onMap(app.maps.activeMap);
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
