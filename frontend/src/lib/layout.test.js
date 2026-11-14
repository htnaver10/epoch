import test from "node:test";
import assert from "node:assert/strict";
import { maxScaleFor } from "./layout.js";

test("the presentation scale stops where the window can no longer lay the console out", () => {
  assert.equal(maxScaleFor({ w: 1440, h: 900 }), 1.2);
  assert.equal(maxScaleFor({ w: 1920, h: 1080 }), 1.3);
  assert.equal(maxScaleFor({ w: 1280, h: 720 }), 1.1);
  assert.equal(maxScaleFor({ w: 2560, h: 1440 }), 1.3);
  assert.equal(maxScaleFor({ w: 390, h: 844 }), 1); // a phone never scales up
  assert.equal(maxScaleFor({ w: 1366, h: 650 }), 1.1);
  assert.equal(maxScaleFor({ w: 1366, h: 560 }), 1); // too short to grow at all
});
