import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { aggregateCells, alertCell, cellAt, cellColor, overviewExport, overviewPoints, overviewURL, overviewView } from "./overview.js";
import { COLORS } from "./data.js";

const records = [
  { date: "2024-06-05", split: "train", group: "west", cells: [[52.125,-125.125,12,3,2,4,1,0,0.8,1]] },
  { date: "2024-06-08", split: "test", group: "west", cells: [[52.125,-125.125,8,2,3,1,0,1,0.7,1], [53.125,-126.125,10,0,0,0,0,0,0.1,1]] },
];
test("Overview sums cached samples across dates and keeps dataset/region filters explicit", () => {
  const cells = aggregateCells(records);
  assert.equal(cells.length,2); assert.equal(cells[0].queries,20); assert.equal(cells[0].score,0.8);
  assert.deepEqual(cells[0].dates,["2024-06-05","2024-06-08"]);
  assert.equal(aggregateCells(records,{split:"test"})[0].queries,8);
  assert.equal(aggregateCells(records,{date:"2024-06-05"}).length,1);
  assert.equal(aggregateCells(records,{region:"east"}).length,0);
});
test("Difference shows query disagreements even when total sensor counts match", () => {
  const cell=aggregateCells(records)[0];
  assert.equal(cell.mlp,cell.modis); assert.ok(alertCell(cell,"difference"));
  assert.equal(cellColor(cell,"difference"),COLORS.both);
  const points=overviewPoints(aggregateCells(records),"difference",false);
  assert.equal(points.length,1); assert.equal(points[0].color,COLORS.both);
  assert.equal(overviewPoints(aggregateCells(records),"mlp",true).length,3);
});
test("Coarse-cell picking and downloads preserve display-grid semantics and provenance", () => {
  const cells=aggregateCells(records),cell=cellAt(cells,52.12,-125.12,0.25);
  assert.equal(cell.key,cells[0].key); assert.equal(cellAt(cells,60,-80,0.25),null);
  const output=overviewExport(cells,{model:{version:"v1"},grid_degrees:0.25,scope:"sampled",count_semantics:"samples may repeat"},{layer:"mlp"});
  assert.equal(output.features.length,1); assert.equal(output.model_version,"v1");
  assert.deepEqual(output.features[0].geometry.coordinates,[-125.125,52.125]);
  assert.deepEqual(output.features[0].properties.dates,cell.dates);
});
test("Overview links preserve date, dataset, region, layer and coverage", () => {
  const view={date:"2024-06-05",split:"test",region:"west",layer:"difference",coverage:false};
  assert.deepEqual(overviewView(overviewURL(view)),view);
  assert.match(overviewURL(view),/view=overview/);
  assert.equal(overviewView("?layer=unknown&split=unknown").split,"all");
});
test("Bundled overview reconciles all 22.2 million source samples without expanding display density", () => {
  const data=JSON.parse(readFileSync(new URL("../../public/data/ml-overview.json",import.meta.url),"utf8"));
  const cells=aggregateCells(data.records);
  assert.equal(cells.length,data.display_cell_count);
  for (const [field,expected] of Object.entries(data.source_totals)) assert.equal(cells.reduce((sum,c) => sum+c[field],0),expected);
  assert.ok(cells.every((c) => Number.isFinite(c.lat) && Number.isFinite(c.lon) && c.queries>0 && c.score>=0 && c.score<=1));
  assert.ok(data.source_totals.queries/cells.length>2000);
});
