import test from "node:test";
import assert from "node:assert/strict";
import { fireRamp, inkOn, C } from "./theme.js";

const luminance = (rgb) => {
  const [r, g, b] = rgb.match(/\d+/g).map((v) => {
    const c = Number(v) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
const hex = (h) => `rgb(${parseInt(h.slice(1, 3), 16)}, ${parseInt(h.slice(3, 5), 16)}, ${parseInt(h.slice(5, 7), 16)})`;

test("the fire ramp gets steadily lighter, so it reads without colour vision", () => {
  let last = -1;
  for (let i = 0; i <= 100; i++) {
    const l = luminance(fireRamp(i / 100));
    assert.ok(l > last, `not lighter at ${i / 100}`);
    last = l;
  }
  assert.ok(luminance(fireRamp(1)) / luminance(fireRamp(0)) > 50); // a wide range of lightness to work with
});

test("ink on a ramp colour always clears 4.5:1", () => {
  for (let i = 0; i <= 100; i++) {
    const fill = fireRamp(i / 100), ink = inkOn(fill);
    const ratio = contrast(luminance(fill), luminance(hex(ink)));
    assert.ok(ratio >= 4.5, `${fill} with ${ink}: ${ratio.toFixed(2)}`);
  }
});

test("the interface accent, the sensors and the verdict tones are different hues", () => {
  const hues = [C.accent, C.viirs, C.high, C.low, C.typical, C.warn];
  assert.equal(new Set(hues).size, hues.length);
  assert.notEqual(C.accent, C.viirs); // blue is the interface, never a sensor
});
