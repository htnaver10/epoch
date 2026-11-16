"""Serve the compiled EPOCH frontend and its API from one local origin."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from main import app as api, lifespan as api_lifespan

SITE = Path(__file__).resolve().parents[1] / "ready-to-run"


@asynccontextmanager
async def lifespan(app):
    # Mounted applications do not receive their own lifespan events.
    async with api_lifespan(api):
        yield


app = FastAPI(title="EPOCH local viewer", lifespan=lifespan)


@app.get("/viewer-health")
def health():
    cache = api.state.ml_catalogue
    return {
        "service": "epoch-local-viewer",
        "model_version": cache.model()["version"] if cache.catalogue else None,
        "scene_count": len(cache.list_scenes()) if cache.catalogue else 0,
        "cache_ready": cache.catalogue is not None,
    }


app.mount("/api", api)
app.mount("/", StaticFiles(directory=SITE, html=True), name="frontend")
