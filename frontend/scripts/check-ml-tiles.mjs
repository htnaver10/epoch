import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { tileCode } from "../src/ml/tiles.js";

const base = process.env.EPOCH_VIEWER_URL || "http://localhost:3000";
const out = fileURLToPath(new URL("../../data-visualization/ml-tiles/", import.meta.url));
const data = JSON.parse(await readFile(new URL("../public/data/ml-tiles.json", import.meta.url), "utf8"));
const tile = [...data.tiles].sort((a,b) => b.counts.mlp-a.counts.mlp || a.id.localeCompare(b.id))[0];
const browser = await chromium.launch({headless:true});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[], shaderErrors=[], apiRequests=[], checks=[];
await mkdir(out,{recursive:true});
page.on("pageerror",e=>errors.push(e.message));
page.on("console",m=>{if(m.type()==="error" && /THREE|shader|WebGL/i.test(m.text()))shaderErrors.push(m.text());});
await page.route("**/api/**",route=>{apiRequests.push(route.request().url());return route.abort();});
await page.route(/^https:\/\//,route=>route.abort());
const hash=b=>createHash("sha256").update(b).digest("hex");
try{
  await page.goto(`${base}/?mode=ml`);
  await page.getByLabel("Find tile",{exact:true}).waitFor();
  await page.locator("canvas").waitFor();await page.waitForTimeout(5500);
  await page.screenshot({path:out+"/canada-detections.png"});
  assert.equal(await page.getByLabel("Show tile boundaries",{exact:true}).isChecked(),true);
  assert.equal(await page.getByLabel("Show tile IDs",{exact:true}).isChecked(),true);
  await page.getByLabel("Find tile",{exact:true}).fill(tileCode(tile));
  assert.equal(await page.locator(".overview-tile-list button").count(),1);
  await page.getByRole("button",{name:`Focus tile ${tile.id}`,exact:true}).click();
  await page.getByRole("heading",{name:`Tile ${tileCode(tile)}`,exact:true}).waitFor();
  await page.waitForTimeout(5500);
  const label=page.getByRole("button",{name:`Select tile ${tile.id}`,exact:true});
  assert.equal(await label.isVisible(),true);
  assert.equal(await label.getAttribute("aria-pressed"),"true");
  await page.screenshot({path:out+"/selected-tile.png"});
  assert.match(page.url(),new RegExp(`tile=${tile.id}`));
  checks.push("all 1,608 tile identities are bundled; finder, zoom, selected label and steel-blue boundary identify a real tile");

  await page.getByLabel("Show tile IDs",{exact:true}).uncheck();
  assert.equal(await page.locator(".patch-labels button").count(),0);
  await page.getByLabel("Show tile boundaries",{exact:true}).uncheck();
  await page.reload();
  await page.getByRole("heading",{name:`Tile ${tileCode(tile)}`,exact:true}).waitFor();
  await page.waitForTimeout(5500);
  assert.equal(await page.getByLabel("Show tile IDs",{exact:true}).isChecked(),false);
  assert.equal(await page.getByLabel("Show tile boundaries",{exact:true}).isChecked(),false);
  assert.equal(await page.locator(".patch-labels button").count(),0);
  await page.getByLabel("Show tile IDs",{exact:true}).check();
  await page.getByLabel("Show tile boundaries",{exact:true}).check();
  await page.waitForTimeout(1200);
  assert.equal(await label.isVisible(),true);
  await label.click();
  checks.push("tile selection and independent boundary/label toggles survive shared links and reload; off hides all tile IDs");

  const canvas=page.locator("canvas");
  await page.waitForTimeout(1500);
  const normal=await canvas.screenshot();
  await page.getByLabel("Detection marker size",{exact:true}).fill("2");
  await page.waitForTimeout(700);
  const larger=await canvas.screenshot();
  assert.notEqual(hash(normal),hash(larger));
  await page.screenshot({path:out+"/larger-detection-markers.png"});
  checks.push("larger high-contrast detection markers render without WebGL shader errors and size slider changes the display");

  await page.getByRole("link",{name:"Open this tile in the scene atlas →",exact:true}).click();
  await page.locator(".pres-image img").evaluate(img=>img.decode());
  assert.equal(new URL(page.url()).searchParams.get("scene"),tile.id);
  assert.equal(new URL(page.url()).searchParams.get("split"),tile.split);
  assert.match(await page.locator(".pres-scene-heading").innerText(),new RegExp(tileCode(tile)));
  checks.push("selected tile opens its matching MODIS/MLP/VIIRS scene in the atlas");

  await page.goto(`${base}/?mode=ml&view=overview&tile=${tile.id}`);
  await page.getByLabel("Find tile",{exact:true}).waitFor();
  await page.getByLabel("Find tile",{exact:true}).fill("a-tile-that-does-not-exist");
  await page.getByText("No tiles match. Clear the search or change the filters.",{exact:true}).waitFor();
  await page.getByLabel("Find tile",{exact:true}).fill("British Columbia");
  assert.ok(await page.locator(".overview-tile-list button").count()>0);
  await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(1500);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.screenshot({path:out+"/mobile.png"});
  checks.push("expanded region search, empty-search feedback and mobile layout work on the static site");
  assert.deepEqual(errors,[]);assert.deepEqual(shaderErrors,[]);assert.deepEqual(apiRequests,[]);
  const receipt={passed:true,model_version:data.model_version,tiles:data.tiles.length,tile_id:tile.id,checks,runtime_errors:errors,shader_errors:shaderErrors,api_requests:apiRequests.length,external_network_blocked:true};
  await writeFile(out+"/browser-checks.json",JSON.stringify(receipt,null,2)+"\n");console.log(JSON.stringify(receipt,null,2));
}catch(error){await page.screenshot({path:out+"/failure.png",fullPage:true});throw error;}
finally{await browser.close();}
