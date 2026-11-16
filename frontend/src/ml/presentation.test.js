import test from "node:test";
import assert from "node:assert/strict";
import { filterScenes, presentationView, selectScene, viewURL } from "./presentation.js";

const scenes = [
  { id: "a", split: "test", date: "2024-06-08", observed_at: "2024-06-08T00:00Z", group: "east", counts: { mlp: 5, viirs: 2 } },
  { id: "b", split: "train", date: "2021-07-11", observed_at: "2021-07-11T00:00Z", group: "central", counts: { mlp: 100, viirs: 80 } },
  { id: "c", split: "test", date: "2024-06-13", observed_at: "2024-06-13T00:00Z", group: "east", counts: { mlp: 2, viirs: 5 } },
];

test("Presentation defaults keep training observations out of the benchmark", () => {
  const initial = presentationView("");
  assert.equal(initial.split, "test");
  assert.deepEqual(filterScenes(scenes, initial.split, "", "", "reference").map((s) => s.id), ["c", "a"]);
  assert.equal(scenes[0].id, "a", "sorting cannot mutate the catalogue");
});

test("Shared scene links preserve the dataset split and rendered layer", () => {
  assert.deepEqual(presentationView(viewURL("b", "train", "difference")), { scene: "b", split: "train", layer: "difference" });
  assert.equal(presentationView("?split=bad&layer=bad").split, "test");
  assert.equal(presentationView("?split=bad&layer=bad").layer, "compare");
});

test("Date/search filters cannot silently retain a scene from another split", () => {
  const matches = filterScenes(scenes, "test", "2024-06-08", "EAST", "alerts");
  assert.equal(selectScene(matches, "b").id, "a");
  assert.equal(selectScene(filterScenes(scenes, "test", "", "missing", "date"), "a"), null);
  assert.equal(filterScenes(scenes, "all", "", "", "alerts")[0].id, "b");
});
