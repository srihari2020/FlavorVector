from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.engine import FlavorEngine
from app.schemas import (
    PairingRequest,
    PairingResponse,
    SubstituteRequest,
    SubstituteResponse,
)

engine: FlavorEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = FlavorEngine()
    yield
    engine = None


app = FastAPI(
    title="FlavorVector API",
    description="Semantic Cooking Substitution & Flavor-Pairing Engine",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": engine.device,
        "substitutions_indexed": engine.sub_collection.count(),
        "pairings_indexed": engine.pair_collection.count(),
    }


@app.post("/substitute", response_model=SubstituteResponse)
def substitute(req: SubstituteRequest):
    results = engine.search_substitutes(
        query=req.query,
        dish_context=req.dish_context,
        vegan_only=req.vegan_only,
        top_k=req.top_k,
    )
    return SubstituteResponse(query=req.query, results=results)


@app.post("/pair", response_model=PairingResponse)
def pair(req: PairingRequest):
    suggestions = engine.suggest_pairings(
        ingredients=req.ingredients,
        dish_style=req.dish_style,
        top_k=req.top_k,
    )
    return PairingResponse(input_ingredients=req.ingredients, suggestions=suggestions)
