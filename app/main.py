from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.balancer import scale_substitution_text
from app.engine import FlavorEngine
from app.schemas import (
    BridgeRequest,
    BridgeResponse,
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
    description="Real-Time Semantic Cooking Substitution & Flavor-Pairing Engine",
    version="1.1.0",
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
    results, fetched_live = engine.search_substitutes(
        query=req.query,
        dish_context=req.dish_context,
        taste_filter=req.taste_filter,
        cuisine_filter=req.cuisine_filter,
        vegan_only=req.vegan_only,
        force_live=req.force_live,
        top_k=req.top_k,
    )
    if req.batch_scale != 1.0:
        for item in results:
            item["substitute"] = scale_substitution_text(item["substitute"], req.batch_scale)
    return SubstituteResponse(query=req.query, fetched_live=fetched_live, results=results)


@app.post("/pair", response_model=PairingResponse)
def pair(req: PairingRequest):
    suggestions, fetched_live = engine.suggest_pairings(
        ingredients=req.ingredients,
        dish_style=req.dish_style,
        target_taste=req.target_taste,
        cuisine_filter=req.cuisine_filter,
        force_live=req.force_live,
        top_k=req.top_k,
    )
    return PairingResponse(
        input_ingredients=req.ingredients,
        fetched_live=fetched_live,
        suggestions=suggestions,
    )


@app.post("/bridge", response_model=BridgeResponse)
def bridge(req: BridgeRequest):
    bridges = engine.find_flavor_bridges(
        pantry=req.pantry,
        dish_goal=req.dish_goal,
        top_k=req.top_k,
    )
    return BridgeResponse(pantry=req.pantry, bridges=bridges)
