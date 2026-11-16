import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { filterTiles, tileBounds, tileCode, tileTitle, tilesAt, tileViewLink } from "./tiles.js";

const tiles=[{id:"abcd1234567890123456",date:"2024-06-05",split:"test",group:"bc_2024",bounds:[-125,51,-124,52],counts:{mlp:3}},
  {id:"dcba9876543210987654",date:"2024-06-08",split:"train",group:"nwt_2024",bounds:[-126,50,-124,53],counts:{mlp:7}}];
test("Tile finder identifies full/short IDs, expanded region names, dates and dataset filters",()=>{
  assert.equal(tileCode(tiles[0]),"abcd1234");assert.match(tileTitle(tiles[0]),/British Columbia/);
  assert.deepEqual(filterTiles(tiles,{},"british columbia").map(t=>t.id),[tiles[0].id]);
  assert.equal(filterTiles(tiles,{split:"test"},"abcd1234").length,1);
  assert.equal(filterTiles(tiles,{date:"2024-06-08",region:"nwt_2024"})[0].id,tiles[1].id);
  assert.equal(filterTiles(tiles,{},"missing").length,0);
  assert.equal(tiles[0].id,"abcd1234567890123456");
});
test("Tile selection keeps overlapping observations distinct and atlas links retain the split",()=>{
  assert.equal(tilesAt(tiles,51.5,-124.5).length,2);
  assert.equal(tilesAt(tiles,80,-80).length,0);
  assert.deepEqual(tileBounds(tiles[0]),[[51,-125],[52,-124]]);
  const query=new URLSearchParams(tileViewLink(tiles[0]));
  assert.equal(query.get("scene"),tiles[0].id);assert.equal(query.get("split"),"test");
});
test("Bundled tile identities match the exact frozen catalogue and have unique short labels",()=>{
  const tiles=JSON.parse(readFileSync(new URL("../../public/data/ml-tiles.json",import.meta.url),"utf8"));
  const overview=JSON.parse(readFileSync(new URL("../../public/data/ml-overview.json",import.meta.url),"utf8"));
  assert.equal(tiles.model_version,overview.model.version);
  assert.equal(tiles.source_catalogue_sha256,overview.source_catalogue_sha256);
  assert.equal(tiles.tiles.length,1608);
  assert.equal(new Set(tiles.tiles.map(tileCode)).size,1608);
  assert.ok(tiles.tiles.every(t=>t.bounds.length===4&&t.bounds.every(Number.isFinite)));
});
