# Integrated MLP viewer verification

These captures show the compiled EPOCH app served by `backend/viewer.py`, using
the real frozen `plswork-mlp-35535575bd95` cache.

- [Desktop comparison](desktop.png)
- [Selected query, score, and sensor flags](query-inspector.png)
- [Mobile layout](mobile.png)
- [Clearly labelled snapshot during API outage](api-offline-snapshot.png)
- [Downloaded MLP-layer GeoJSON](selected-scene.geojson)
- [Automated browser receipt](browser-checks.json)

The browser receipt records ten passed checks and zero runtime errors. It also
tests all three dates, source layer counts, disagreement filtering, accessible
inspection, shareable links, a scene without alerts, navigation to Calendar and
Scenarios, API failure, and recovery when both data sources initially fail.

Additional verification: 100 frontend unit tests, six backend integration tests,
frontend lint, and the production build pass. API checks reconcile all 84 patches
and 585,464 reference-matched queries with the saved policy: TP 70, FP 72, FN 25.
No new training or satellite download was performed.

Run instructions, API routes, source provenance, and research limitations are in
[MLP-INTEGRATION.md](../../MLP-INTEGRATION.md).
