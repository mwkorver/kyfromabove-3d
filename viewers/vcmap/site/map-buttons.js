/* The map switch at the top left says what each map is: "3D (Cesium)" and
 * "Obliques (OpenLayers)" written in place of their icons (the 3D icon is the
 * letters "3D", which would say it twice). The 2D map's button is
 * hidden but the map stays in the config: the background imagery reaches the
 * 3D map only while the 2D map is there. VC Map draws its toolbar buttons as
 * icons only, so the words are added with a style rule on each button's
 * data-action-name, which outlasts the toolbar redrawing itself.
 */
(function mapButtons() {
  const labels = { CesiumMap: '3D (Cesium)', ObliqueMap: 'Obliques (OpenLayers)' };
  const rules = Object.entries(labels).map(([map, text]) => `
    .vcs-navbar button[data-action-name="${map}"] { width: auto; padding: 0 10px; text-transform: none; }
    .vcs-navbar button[data-action-name="${map}"] .v-icon { display: none; }
    .vcs-navbar button[data-action-name="${map}"] .v-btn__content::after {
      content: "${text}"; font-size: 13px; letter-spacing: normal;
    }`);
  rules.push('.vcs-navbar button[data-action-name="OpenlayersMap"] { display: none !important; }');
  const style = document.createElement('style');
  style.id = 'kyfa-map-buttons';
  style.textContent = rules.join('\n');
  document.head.appendChild(style);
})();
