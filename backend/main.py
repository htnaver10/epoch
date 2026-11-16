from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import os

from api.routes import router
from api.ml_routes import router as ml_router
from services.ml_catalogue import MLCatalogue

@asynccontextmanager
async def lifespan(app):
    cache=os.getenv('EPOCH_ML_CACHE_DIR',str(Path(__file__).parent/'data/ml_cache'))
    app.state.ml_catalogue=MLCatalogue(cache)
    yield
    app.state.ml_catalogue=None

app = FastAPI(title="EPOCH API",lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ml_router)


@app.get("/")
def home():
    return {"message": "EPOCH API is running"}
