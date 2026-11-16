import test from "node:test";
import assert from "node:assert/strict";
import { visibleFeatures, researchPoints, readMLView, writeMLView, validateScene, nearestFeature, geoJSONDownload, COLORS } from "./data.js";
const feature = (id, m, s, v, x = -75) => ({ id, type: "Feature", geometry: { type: "Point", coordinates: [x, 52] }, properties: { mlp_detected: m, modis_detected: s, viirs_reference: v, mlp_score: 0.3 } });
const features = [feature("both",true,true,true),feature("mlp",true,false,false),feature("modis",false,true,false),feature("viirs",false,false,true)];
test("Research layers filter each source and compare disagreement without relabeling VIIRS", () => {
  assert.deepEqual(visibleFeatures(features,"mlp").map((f) => f.id),["both","mlp"]);
  assert.deepEqual(visibleFeatures(features,"difference").map((f) => f.id),["mlp","modis"]);
  assert.deepEqual(visibleFeatures(features,"viirs").map((f) => f.id),["both","viirs"]);
});
test("Coverage has its own marks and cannot become alerts", () => {
  const points=researchPoints(features,[[-76,53]],"mlp",true);
  assert.equal(points.length,3);assert.equal(points[0].color,COLORS.coverage);assert.equal(points[1].color,COLORS.mlp);
  assert.equal(researchPoints(features,[[-76,53]],"mlp",false).length,2);
});
test("ML view links preserve scene, layer and coverage; unsupported layers are rejected", () => {
  assert.deepEqual(readMLView(writeMLView("abc","difference",false)),{scene:"abc",layer:"difference",coverage:false});
  assert.equal(readMLView("?layer=unknown").layer,"all");
});
test("Scene responses cannot mix versions or invalid coordinates", () => {
  const data={type:"FeatureCollection",scene:{id:"abc",model_version:"v1"},features};
  const coverage={properties:{scene_id:"abc",count:1},geometry:{type:"MultiPoint",coordinates:[[-75,52]]}};
  assert.equal(validateScene(data,coverage,"abc","v1").coverage.length,1);
  assert.throws(() => validateScene(data,coverage,"abc","v2"));
  assert.throws(() => validateScene(data,{...coverage,geometry:{type:"MultiPoint",coordinates:[[NaN,52]]}},"abc","v1"));
});
test("Picking returns nearby rendered features, never a distant alert", () => {
  assert.equal(nearestFeature(features,52,-75).id,"both");assert.equal(nearestFeature(features,60,-100),null);
});
test("GeoJSON export retains model provenance, layer semantics and score values", () => {
  const data={scene:{model_version:"v1",observed_at:"2024-06-13T17:40:00Z",coverage_kind:"reference_matched_query_centers"},features};
  const result=geoJSONDownload(data,"mlp");assert.equal(result.features.length,2);assert.equal(result.model_version,"v1");assert.match(result.note,/unavailable/);
});
