import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { aggregateCells, alertCell } from "../src/ml/overview.js";

const base = process.env.EPOCH_VIEWER_URL || "http://localhost:3000";
const out = fileURLToPath(new URL("../../data-visualization/ml-overview/", import.meta.url));
const data = JSON.parse(await readFile(new URL("../public/data/ml-overview.json", import.meta.url), "utf8"));
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [], apiRequests = [], checks = [];
await mkdir(out, { recursive: true });
page.on("pageerror", (e) => errors.push(e.message));
await page.route("**/api/**", (route) => { apiRequests.push(route.request().url()); return route.abort(); });
await page.route(/^https:\/\//, (route) => route.abort());
const ready = async () => {
  await page.getByText("Bundled data · no backend", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Download overview GeoJSON", exact: true }).waitFor();
};
try {
  await page.goto(`${base}/?mode=ml`);
  await ready(); await page.locator("canvas").waitFor();
  await page.waitForTimeout(5500);
  assert.equal(await page.getByText("This browser could not start WebGL.", { exact: false }).count(), 0);
  await page.screenshot({ path: out + "/canada-overview.png" });
  const all = aggregateCells(data.records);
  assert.equal(all.length, data.display_cell_count);
  assert.equal(await page.locator(".ml-accessible tbody tr").count(), all.filter((c) => alertCell(c, "all")).length);
  assert.deepEqual(apiRequests, []);
  checks.push("real sparse 50.6M overview renders in WebGL from npx serve with no API or external network");

  for (const [layer, label] of [["mlp", "50.6M MLP"], ["modis", "MODIS"], ["viirs", "VIIRS reference"], ["difference", "Difference"]]) {
    await page.getByRole("button", { name: label, exact: true }).click();
    assert.equal(await page.locator(".ml-accessible tbody tr").count(), all.filter((c) => alertCell(c, layer)).length);
  }
  await page.locator(".ml-accessible summary").click();
  await page.getByRole("button", { name: /^Inspect cell/ }).first().click();
  await page.getByRole("heading", { name: "Sampled display cell", exact: true }).waitFor();
  await page.getByRole("button", { name: "Close overview selection" }).click();
  const event = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download overview GeoJSON", exact: true }).click();
  await (await event).saveAs(out + "/overview.geojson");
  const exported = JSON.parse(await readFile(out + "/overview.geojson", "utf8"));
  assert.equal(exported.model_version, data.model.version);
  assert.equal(exported.grid_degrees, 0.25);
  assert.equal(exported.features.length, all.filter((c) => alertCell(c, "difference")).length);
  checks.push("every source layer, query-level disagreement aggregation, accessible inspection and GeoJSON export");

  await page.getByLabel("Overview dataset", { exact: true }).selectOption("test");
  await page.getByLabel("Overview date", { exact: true }).selectOption("2024-06-08");
  const filtered = aggregateCells(data.records, { split: "test", date: "2024-06-08" });
  assert.equal(await page.locator(".ml-accessible tbody tr").count(), filtered.filter((c) => alertCell(c, "difference")).length);
  await page.getByLabel("Show cells containing cached samples").uncheck();
  await page.reload(); await ready();
  assert.equal(await page.getByLabel("Overview date", { exact: true }).inputValue(), "2024-06-08");
  assert.equal(await page.getByLabel("Show cells containing cached samples").isChecked(), false);
  await page.getByLabel("Overview region", { exact: true }).selectOption("bc_2021");
  await page.getByText("No displayed alerts in the selected cached samples.", { exact: true }).waitFor({ state: "attached" });
  await page.getByLabel("Overview date", { exact: true }).selectOption("");
  await page.getByLabel("Overview dataset", { exact: true }).selectOption("all");
  await page.getByRole("button", { name: "Compare", exact: true }).click();
  await page.getByLabel("Show cells containing cached samples").check();
  await page.getByRole("button", { name: "Fit samples", exact: true }).click();
  await page.waitForTimeout(5500);
  checks.push("date, dataset and region filters, empty-state honesty, shared links and reload");

  // The globe tapers horizon tilt at orbital height. Enter its regional zoom
  // range before testing the near-ground 3D toggle.
  for (let i = 0; i < 3; i++) {
    await page.getByRole("button", { name: "Zoom in", exact: true }).click();
    await page.waitForTimeout(1000);
  }
  const canvas = page.locator("canvas"), hash = (b) => createHash("sha256").update(b).digest("hex");
  const flat = await canvas.screenshot();
  await page.getByRole("button", { name: "Tilt the view", exact: true }).click();
  await page.waitForTimeout(1800);
  const tilted = await canvas.screenshot();
  await page.screenshot({ path: out + "/regional-3d.png" });
  assert.notEqual(hash(flat), hash(tilted));
  await page.getByRole("button", { name: "Tilt the view", exact: true }).click();
  await page.waitForTimeout(1800);
  assert.notEqual(hash(tilted), hash(await canvas.screenshot()));
  checks.push("3D tilt and return both change the rendered geographic view");

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(1000);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: out + "/mobile.png" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  checks.push("mobile layout without horizontal page overflow");

  await page.getByRole("link", { name: "Scene atlas", exact: true }).click();
  await page.getByRole("heading", { name: "Scene comparison", exact: true }).waitFor();
  await page.locator(".pres-image img").evaluate((img) => img.decode());
  await page.getByRole("link", { name: "3D globe", exact: true }).click(); await ready();
  await page.getByRole("button", { name: "Calendar", exact: true }).click();
  await page.getByRole("button", { name: "Scenarios", exact: true }).click();
  await page.getByRole("button", { name: "MLP research", exact: true }).waitFor();
  assert.match(page.url(), /mode=scenarios/);
  checks.push("static atlas, sparse overview, Calendar and Scenarios remain reachable");

  await page.route("**/data/ml-overview.json", (route) => route.fulfill({ status: 503, body: "Unavailable" }));
  await page.goto(`${base}/?mode=ml&view=overview`);
  await page.getByText("Overview unavailable", { exact: true }).waitFor();
  await page.unroute("**/data/ml-overview.json");
  await page.getByRole("button", { name: "Retry", exact: true }).click(); await ready();
  checks.push("missing bundled overview produces a recoverable error");

  assert.deepEqual(errors, []); assert.deepEqual(apiRequests, []);
  const receipt = { passed: true, static_server: "npx serve", model_version: data.model.version, source_queries: data.source_totals.queries,
    display_cells: data.display_cell_count, api_requests: apiRequests.length, external_network_blocked: true, checks, runtime_errors: errors };
  await writeFile(out + "/browser-checks.json", JSON.stringify(receipt, null, 2) + "\n");
  console.log(JSON.stringify(receipt, null, 2));
} catch (error) {
  await page.screenshot({ path: out + "/failure.png", fullPage: true });
  throw error;
} finally { await browser.close(); }
