// Run against the built local server; exercise real cached scenes, not mock metrics.
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const base = process.env.EPOCH_VIEWER_URL || "http://127.0.0.1:8000";
const out = fileURLToPath(new URL("../../data-visualization/mlp-integration/", import.meta.url));
const catalogue = JSON.parse(await readFile(new URL("../public/data/ml/catalog.json", import.meta.url), "utf8"));
const richest = [...catalogue.scenes].sort((a,b) => b.counts.mlp - a.counts.mlp)[0];
const empty = catalogue.scenes.find((s) => !s.counts.mlp && !s.counts.modis && !s.counts.viirs);
const checks = [], errors = [];
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
page.on("pageerror", (e) => errors.push(e.message));
const ready = async (p) => {
  await p.getByText("Backend cache", { exact: true }).waitFor();
  await p.getByRole("button", { name: "Download GeoJSON", exact: true }).waitFor();
};
try {
  await page.goto(`${base}/?mode=ml&view=inspect`);
  await ready(page);
  assert.equal(await page.getByLabel("Scene patch", { exact: true }).inputValue(), richest.id);
  await page.locator("canvas").waitFor();
  await page.waitForTimeout(5200); // allow the globe's opening camera flight to complete
  assert.equal(await page.getByText("This browser could not start WebGL.", { exact: false }).count(), 0);
  await page.screenshot({ path: out + "/desktop.png" });
  checks.push("real backend catalogue, default scene and WebGL canvas");

  await page.getByRole("button", { name: "MLP", exact: true }).click();
  await page.locator(".ml-accessible summary").click();
  assert.equal(await page.locator(".ml-accessible tbody tr").count(), richest.counts.mlp);
  await page.getByRole("button", { name: /^Inspect query/ }).first().click();
  await page.getByRole("heading", { name: "Selected query", exact: true }).waitFor();
  await page.screenshot({ path: out + "/query-inspector.png" });
  await page.getByRole("button", { name: "Close selected prediction" }).click();
  checks.push("MLP layer and accessible query inspection");

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download GeoJSON" }).click();
  const download = await downloadPromise;
  await download.saveAs(out + "/selected-scene.geojson");
  const exported = JSON.parse(await readFile(out + "/selected-scene.geojson", "utf8"));
  assert.equal(exported.features.length, richest.counts.mlp);
  assert.equal(exported.model_version, catalogue.model.version);
  checks.push("GeoJSON download preserves actual selected features and model version");

  for (const [name, count] of [["MODIS", richest.counts.modis], ["VIIRS reference", richest.counts.viirs]]) {
    await page.getByRole("button", { name, exact: true }).click();
    assert.equal(await page.locator(".ml-accessible tbody tr").count(), count);
  }
  checks.push("MODIS and VIIRS reference layer counts match the source catalogue");

  await page.getByRole("button", { name: "MLP / MODIS difference", exact: true }).click();
  assert.equal(await page.locator(".ml-accessible tbody tr").count(), richest.counts.mlp_only + richest.counts.modis_only);
  await page.getByLabel("Show cached query coverage").uncheck();
  assert.match(page.url(), /coverage=0/);
  for (const date of [...new Set(catalogue.scenes.map((s) => s.date))]) {
    await page.getByLabel("Date (UTC)", { exact: true }).selectOption(date);
    await ready(page);
    const id = await page.getByLabel("Scene patch", { exact: true }).inputValue();
    assert.equal(catalogue.scenes.find((s) => s.id === id).date, date);
  }
  checks.push("difference filter, coverage toggle, all three dates");

  assert.ok(empty, "benchmark must include a no-alert scene for the empty-state check");
  await page.goto(`${base}/?mode=ml&view=inspect&scene=${empty.id}&layer=mlp&coverage=0`);
  await ready(page);
  await page.locator(".ml-accessible summary").click();
  await page.getByText("No alerts for this layer in this cached patch.", { exact: true }).waitFor();
  assert.equal(await page.getByLabel("Scene patch", { exact: true }).inputValue(), empty.id);
  checks.push("shareable scene link and no-alert state");

  await page.goto(`${base}/?mode=ml&view=inspect&scene=${richest.id}&layer=all`);
  await ready(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(5500);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "mobile page overflows horizontally");
  await page.screenshot({ path: out + "/mobile.png" });
  checks.push("390px mobile layout without horizontal page overflow");
  await page.setViewportSize({ width: 1440, height: 1000 });

  await page.getByRole("button", { name: "Calendar", exact: true }).click();
  await page.getByRole("button", { name: "MLP research", exact: true }).waitFor();
  await page.getByRole("button", { name: "Scenarios", exact: true }).click();
  await page.getByRole("button", { name: "MLP research", exact: true }).waitFor();
  assert.match(page.url(), /mode=scenarios/);
  await page.getByRole("button", { name: "MLP research", exact: true }).click();
  await page.getByRole("heading", { name: "See the comparison." }).waitFor();
  await page.goto(`${base}/?mode=ml&view=inspect`);
  await ready(page);
  checks.push("Calendar / Scenarios / MLP navigation");

  await page.route("**/api/ml/**", (route) => route.abort());
  await page.goto(`${base}/?mode=ml&view=inspect`);
  await page.getByText("Bundled snapshot · API unavailable", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Download GeoJSON" }).waitFor();
  await page.screenshot({ path: out + "/api-offline-snapshot.png" });
  checks.push("API outage uses explicitly labelled same-version bundled snapshot");

  await page.route("**/data/ml/catalog.json", (route) => route.fulfill({ status: 503, body: "Unavailable" }));
  await page.reload();
  await page.getByText("Scene unavailable", { exact: true }).waitFor();
  await page.unroute("**/data/ml/catalog.json");
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await page.getByRole("button", { name: "Download GeoJSON" }).waitFor();
  checks.push("complete cache outage shows unavailable; Retry recovers");

  assert.deepEqual(errors, [], "browser runtime errors");
  const report = { passed: true, model_version: catalogue.model.version, scenes: catalogue.scenes.length, checks, runtime_errors: errors };
  await writeFile(out + "/browser-checks.json", JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  await page.screenshot({ path: out + "/failure.png", fullPage: true });
  throw error;
} finally {
  await browser.close();
}
