from pydantic import BaseModel, Field


class SubstituteRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        examples=["what can replace buttermilk and give the same tanginess in pancakes?"],
    )
    dish_context: str | None = Field(default=None, examples=["fluffy pancakes"])
    taste_filter: str | None = Field(default=None, examples=["sour"])
    cuisine_filter: str | None = Field(default=None, examples=["Baking & Pastry"])
    vegan_only: bool = Field(default=False)
    force_live: bool = Field(default=False)
    top_k: int = Field(default=4, ge=1, le=10)


class SubstituteHit(BaseModel):
    rank: int
    target: str
    substitute: str
    tastes: str
    cuisine: str
    vegan: bool
    source: str
    relevance_score: float
    details: str


class SubstituteResponse(BaseModel):
    query: str
    fetched_live: bool
    results: list[SubstituteHit]


class PairingRequest(BaseModel):
    ingredients: list[str] = Field(..., min_length=1, max_length=6, examples=[["garlic", "lemon", "thyme"]])
    dish_style: str | None = Field(default=None, examples=["roasted chicken"])
    target_taste: str | None = Field(default=None, examples=["cooling"])
    cuisine_filter: str | None = Field(default=None, examples=["Mediterranean & European"])
    force_live: bool = Field(default=False)
    top_k: int = Field(default=6, ge=1, le=15)


class PairingHit(BaseModel):
    rank: int
    ingredient: str
    tastes: str
    cuisine: str
    relevance_score: float
    common_partners: str
    compounds: str
    details: str


class PairingResponse(BaseModel):
    input_ingredients: list[str]
    fetched_live: bool
    suggestions: list[PairingHit]
