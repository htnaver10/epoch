import test from "node:test";
import assert from "node:assert/strict";
import { PLACES, parseCoords, parseDay, interpret, nearestPlace } from "./places.js";

const ctx = { year: 2023, years: [2021, 2022, 2023], dates: ["2023-06-01", "2023-07-14", "2023-08-31"] };

test("coordinates parse in the forms people type", () => {
  assert.deepEqual(parseCoords("58.4 -112.1"), { lat: 58.4, lon: -112.1 });
  assert.deepEqual(parseCoords("58.4N 112.1W"), { lat: 58.4, lon: -112.1 });
  assert.deepEqual(parseCoords("58.4, 112.1"), { lat: 58.4, lon: -112.1 }); // bare longitude is west
  assert.deepEqual(parseCoords("60°n, 25.5°e"), { lat: 60, lon: 25.5 });
  assert.equal(parseCoords("95 10"), null);
  assert.equal(parseCoords("2023"), null);
  assert.equal(parseCoords("fort mcmurray"), null);
});

test("days parse against the open season and reject impossible dates", () => {
  assert.equal(parseDay("jul 14", 2023), "2023-07-14");
  assert.equal(parseDay("14 July", 2023), "2023-07-14");
  assert.equal(parseDay("2021-08-02", 2023), "2021-08-02");
  assert.equal(parseDay("jun 31", 2023), null);
  assert.equal(parseDay("smoke 14", 2023), null);
});

test("queries become runnable actions, most specific first", () => {
  assert.equal(interpret("58.4 -112.1", ctx)[0].type, "coords");
  assert.deepEqual(interpret("2022", ctx)[0], { type: "year", year: 2022, label: "2022", hint: "open season" });
  assert.equal(interpret("1999", ctx).length, 0);
  assert.equal(interpret("jul 14", ctx)[0].date, "2023-07-14");
  assert.equal(interpret("sep 14", ctx).length, 0); // outside the archived season
  assert.equal(interpret("west", ctx)[0].key, "west");
  assert.equal(interpret("peak", ctx)[0].type, "peak");
  assert.equal(interpret("record", ctx)[0].type, "panel");
  const place = interpret("fort mc", ctx)[0];
  assert.equal(place.type, "place");
  assert.equal(place.name, "Fort McMurray");
  assert.equal(interpret("lake", ctx)[0].name, "Lake Athabasca"); // prefix matches lead
  assert.ok(interpret("lake", ctx).length > 2);
  assert.deepEqual(interpret("  ", ctx), []);
});

test("every place is in Canada, and some sit outside the analysis area on purpose", () => {
  for (const p of PLACES) {
    assert.ok(p.lat >= 41.5 && p.lat <= 71 && p.lon >= -142 && p.lon <= -52, p.name);
    assert.ok(p.km >= 100 && p.km <= 600, p.name);
  }
  // their reports say "outside the analysis area", which is more use than "no match"
  const outside = PLACES.filter((p) => p.lat < 50 || p.lon > -60).map((p) => p.name);
  for (const name of ["Vancouver", "Winnipeg", "Toronto", "Montreal", "Halifax", "St. John's"]) assert.ok(outside.includes(name), name);
});

test("well-known centres are found, inside the analysis area or not", () => {
  for (const name of ["calgary", "regina", "kamloops", "toronto", "halifax"]) {
    const hit = interpret(name, ctx)[0];
    assert.equal(hit?.type, "place", name);
    assert.equal(hit.name.toLowerCase(), name);
  }
});

test("a cut result list can be told from a complete one", () => {
  assert.equal(interpret("fort", ctx).length, 7);
  assert.ok(interpret("fort", ctx, 8).length === 8); // more than seven exist
  assert.ok(interpret("fort mc", ctx, 8).length < 8);
});

test("an unnamed point takes the name of the nearest town, within reach", () => {
  assert.equal(nearestPlace(62.5, -114.2).name, "Yellowknife");
  assert.equal(nearestPlace(62.5, -114.2, 5), null); // 11 km away
  assert.equal(nearestPlace(69.5, -100), null); // nothing within 120 km
  assert.ok(["town", "city"].includes(nearestPlace(56.0, -114.5).kind)); // never a province or a lake
  assert.ok(PLACES.length > 90);
});
