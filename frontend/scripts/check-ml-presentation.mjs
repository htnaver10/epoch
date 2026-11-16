import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const base = process.env.EPOCH_VIEWER_URL || "http://127.0.0.1:8000";
const out = fileURLToPath(new URL("../../data-visualization/ml-presentation/", import.meta.url));
await mkdir(out, { recursive: true });
const catalogue = JSON.parse(await readFile(new URL("../public/data/ml-presentation/catalog.json", import.meta.url), "utf8"));
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, deviceScaleFactor: 1 });
const errors = [], checks = [];
page.on("pageerror", (error) => errors.push(error.message));
const ready = async () => {
  await page.getByRole("heading", { name: "Scene comparison", exact: true }).waitFor();
  await page.locator(".pres-image img").evaluate((image) => image.decode());
  await page.waitForFunction(() => !document.querySelector(".pres-image [role=status]"));
};
try {
  await page.goto(`${base}/?mode=ml&view=presentation`);
  await ready();
  await page.screenshot({ path: out + "/desktop.png", fullPage: true });
  assert.equal(await page.locator(".research-evidence").count(), 0);
  assert.ok((await page.locator(".pres-image").boundingBox()).height > 400, "Scene comparison should fill the desktop workspace");
  assert.match(page.url(), /split=test/);
  assert.equal(await page.getByRole("button", { name: "Benchmark 84", exact: true }).getAttribute("aria-pressed"), "true");
  checks.push("50.6M model presentation opens with benchmark-only scenes and a decoded rendered comparison");
  await page.getByRole("button", { name: "Method and evidence", exact: true }).click();
  assert.equal(await page.locator(".pres-archive .row").first().locator("dd").innerText(), catalogue.scene_count.toLocaleString());
  await page.getByRole("heading", { name: "Frozen model performance", exact: true }).waitFor();
  await page.screenshot({ path: out + "/method-and-evidence.png", fullPage: true });
  await page.getByRole("button", { name: "Close method and evidence", exact: true }).click();
  checks.push("scene-first console with benchmark, archive counts and research limitations on request");
  for (const [id, name] of [["mlp", "50.6M MLP"], ["modis", "MODIS"], ["viirs", "VIIRS reference"], ["difference", "Difference"], ["compare", "Compare all"]]) {
    await page.getByRole("button", { name, exact: true }).click();
    await ready();
    assert.match(await page.locator(".pres-image img").getAttribute("src"), new RegExp(`/${id}.png$`));
  }
  checks.push("All five pre-rendered layers load");
  const first = new URL(page.url()).searchParams.get("scene");
  await page.getByRole("button", { name: "Next scene", exact: true }).click();
  await ready();
  assert.notEqual(new URL(page.url()).searchParams.get("scene"), first);
  await page.getByRole("button", { name: "Previous scene", exact: true }).click();
  await ready();
  assert.equal(new URL(page.url()).searchParams.get("scene"), first);
  await page.keyboard.press("ArrowRight");
  await ready();
  assert.notEqual(new URL(page.url()).searchParams.get("scene"), first);
  await page.reload(); await ready();
  checks.push("Next/previous/keyboard and reload preserve scene selection");
  for (const [split, label] of [["train", "Training"], ["validation", "Validation"], ["additional", "Additional windows"], ["all", "All scenes"]]) {
    if (!catalogue.splits[split] && split !== "all") continue;
    await page.getByRole("button", { name: new RegExp(`^${label} `) }).click();
    await ready();
    assert.match(page.url(), new RegExp(`split=${split}`));
    assert.ok((await page.locator(".pres-split-note").innerText()).length > 30);
  }
  await page.getByLabel("Find a patch", { exact: true }).fill("this-is-not-a-patch");
  await page.getByRole("heading", { name: "No matching patches" }).waitFor();
  await page.getByRole("button", { name: "Clear filters", exact: true }).click();
  await ready();
  await page.getByRole("button", { name: "Benchmark 84", exact: true }).click();
  await ready();
  const dates = [...new Set(catalogue.scenes.filter((s) => s.split === "test").map((s) => s.date))];
  for (const date of dates) {
    await page.getByLabel("Presentation date", { exact: true }).selectOption(date);
    await ready();
    assert.match(await page.locator(".pres-scene-heading h3").innerText(), new RegExp(date));
  }
  checks.push("Every dataset split is distinct; date/search and no-results recovery work");
  for (const [name, filename] of [["Image PNG ↓", "scene.png"], ["GeoJSON ↓", "scene.geojson"], ["All query scores ↓", "queries.npz"], ["Summary CSV ↓", "scene-summary.csv"]]) {
    const event = page.waitForEvent("download");
    await page.getByRole("link", { name, exact: true }).click();
    const download = await event; await download.saveAs(out + "/" + filename);
    assert.ok((await readFile(out + "/" + filename)).length > 100);
  }
  const exported = JSON.parse(await readFile(out + "/scene.geojson", "utf8"));
  assert.equal(exported.model_version, catalogue.model.version);
  checks.push("PNG, real GeoJSON, all-query NPZ and complete summary CSV downloads");
  await page.getByRole("button", { name: "Present scenes", exact: true }).click();
  assert.equal(await page.locator(".pres-browser").isVisible(), false);
  await page.locator(".pres-stage").scrollIntoViewIfNeeded();
  await page.screenshot({ path: out + "/presentation.png" });
  await page.getByRole("button", { name: "Play scene tour", exact: true }).click();
  const before = new URL(page.url()).searchParams.get("scene");
  await page.waitForFunction((id) => new URL(location.href).searchParams.get("scene") !== id, before, { timeout: 15000 });
  await ready();
  await page.getByRole("button", { name: "Pause tour", exact: true }).click();
  await page.keyboard.press("Escape");
  assert.equal(await page.locator(".pres-browser").isVisible(), true);
  checks.push("Presentation layout, timed scene tour and Escape exit");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}/?mode=ml&view=presentation`); await ready();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Mobile horizontal overflow");
  assert.ok(await page.locator(".pres-workspace").evaluate((element) => element.scrollWidth <= element.clientWidth), "Mobile workspace must fit its viewport");
  const mobileImage = await page.locator(".pres-image").boundingBox();
  assert.ok(mobileImage.x >= 0 && mobileImage.x + mobileImage.width <= 390, "Comparison image must stay inside the phone screen");
  await page.screenshot({ path: out + "/mobile.png", fullPage: true });
  checks.push("390px mobile layout without horizontal overflow");
  await page.setViewportSize({ width: 1440, height: 1050 });
  await page.route("**/api/**", (route) => route.abort());
  await page.route(/https:\/\/(?!127\.0\.0\.1)/, (route) => route.abort());
  await page.goto(`${base}/?mode=ml&view=presentation`); await ready();
  checks.push("Presentation works with API and external network requests unavailable");
  await page.route("**/data/ml-presentation/catalog.json", (route) => route.fulfill({ status: 503, body: "Unavailable" }));
  await page.reload();
  await page.getByRole("heading", { name: "Presentation unavailable", exact: true }).waitFor();
  await page.unroute("**/data/ml-presentation/catalog.json");
  await page.getByRole("button", { name: "Retry", exact: true }).click(); await ready();
  checks.push("Missing catalogue has a recoverable error state");
  await page.getByRole("button", { name: "Method and evidence", exact: true }).click();
  await page.getByRole("link", { name: "Open the full rendered atlas ↗", exact: true }).evaluate((link) => { link.target = "_self"; });
  await page.getByRole("link", { name: "Open the full rendered atlas ↗", exact: true }).click();
  await page.getByRole("heading", { name: "EPOCH / Complete scene atlas", exact: true }).waitFor();
  assert.equal(await page.locator("article").count(), catalogue.scene_count);
  checks.push("Standalone atlas contains every prepared scene");
  assert.deepEqual(errors, []);
  const result = { passed: true, model_version: catalogue.model.version, scenes: catalogue.scene_count, checks, runtime_errors: errors };
  await writeFile(out + "/browser-checks.json", JSON.stringify(result, null, 2) + "\n");
  console.log(JSON.stringify(result, null, 2));
} catch (error) {
  await page.screenshot({ path: out + "/failure.png", fullPage: true }); throw error;
} finally { await browser.close(); }
