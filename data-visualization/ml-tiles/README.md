# Tile identification and marker visibility

Browser verification runs the compiled demo through `npx serve`, with external
network and API requests blocked. The captures use real bundled observations.

- [Canada overview with brighter detections](canada-detections.png)
- [Selected tile, readable ID, envelope and sensor counts](selected-tile.png)
- [Enlarged detection markers](larger-detection-markers.png)
- [Mobile layout](mobile.png)
- [Verification receipt](browser-checks.json)

The checks exercise tile search, selection and camera focus, exact scene links,
independent label/boundary toggles and reload, the marker-size control, mobile
layout, and WebGL shader errors. The tile index contains all 1,608 scene identities
from the frozen 50.6M model catalogue; it adds no predictions or satellite data.
