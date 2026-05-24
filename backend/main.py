from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""
backend/main.py
================
FastAPI application exposing:

  POST /simulate          Task A — simulate a review + rating for an unseen item
  POST /recommend         Task B — personalised recommendations
  POST /persona           Build/retrieve a user persona
  POST /index/build       Trigger ChromaDB index build from a Parquet file
  GET  /health            Service health check
  GET  /stats             Vector store statistics
"""


import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agents.persona_builder import PersonaBuilder
from backend.agents.recommender import Recommender
from backend.agents.review_simulator import ReviewSimulator, simulate_batch
from backend.core.vector_store import VectorStore, build_index_from_parquet


# ─── App lifespan ──────────────────────────────────────────────────────────────

vs: VectorStore | None = None
pb: PersonaBuilder | None = None
sim: ReviewSimulator | None = None
rec: Recommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vs, pb, sim, rec
    print("Initialising NaijaNutri services…")
    vs  = VectorStore()
    pb  = PersonaBuilder(vs)
    sim = ReviewSimulator(vs, pb)
    rec = Recommender(vs, pb)

    # # Auto-index: use all_reviews.parquet for full item catalogue.
    # # train/test parquets are only for evaluation scripts.
    # default_parquet = "data/processed/all_reviews.parquet"
    # if not os.path.exists(default_parquet):
    #     default_parquet = "data/processed/train.parquet"  # fallback
    # if os.path.exists(default_parquet) and vs.stats()["reviews"] == 0:
    #     print(f"Auto-indexing from {default_parquet}…")
    #     build_index_from_parquet(default_parquet, vs)

    print(f"Ready. Vector store stats: {vs.stats()}")
    yield
    print("Shutting down.")


app = FastAPI(
    title="NaijaNutri Pro API",
    description="LLM-powered user modelling and recommendation for Nigerian users",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ─────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    user_id:       str
    item_id:       str
    item_name:     str
    item_domain:   str = "yelp"
    item_category: str = "Restaurant"
    item_metadata: dict = Field(default_factory=dict)
    apply_nigerian_adapter: bool = True


class SimulateResponse(BaseModel):
    user_id:          str
    item_id:          str
    item_name:        str
    predicted_rating: float
    review_text:      str
    raw_review_text:  str
    reasoning:        str
    archetype_name:   str


class RecommendRequest(BaseModel):
    user_id:              str
    query:                str
    n:                    int = 10
    domains:              Optional[list[str]] = None
    budget_naira:         Optional[int] = None
    # Food-only params
    dietary_restrictions: list[str] = Field(default_factory=list)
    calorie_target:       Optional[int] = None
    meals_per_day:        int = 3
    weight_goal:          str = "maintain"
    location:             str = ""
    # Products-only params
    product_interests:    list[str] = Field(default_factory=list)
    # Books-only params
    book_genres:          list[str] = Field(default_factory=list)
    # Shared
    language_preference:  str = "english"
    conversation_history: list[dict] = Field(default_factory=list)
    plan_mode:            bool = False
    exclude_items:        list[str] = Field(default_factory=list)


class RecommendedItemOut(BaseModel):
    item_id:       str
    item_name:     str
    item_domain:   str
    item_category: str
    avg_rating:    float
    score:         float
    reason:        str
    budget_ok:     bool


class RecommendResponse(BaseModel):
    user_id:           str
    query:             str
    recommendations:   list[RecommendedItemOut]
    archetype_name:    str
    is_cold_start:     bool
    strategy_used:     str
    response_text:     str
    meal_plan:         Optional[list[dict]] = None
    tdee_used:         Optional[int] = None
    calories_per_meal: Optional[int] = None


class PersonaRequest(BaseModel):
    user_id:          str
    force_rebuild:    bool = False
    # For explicit cold-start persona
    archetype_key:    Optional[str] = None
    budget_naira:     Optional[int] = None
    dietary_notes:    str = ""
    liked_categories: Optional[list[str]] = None
    disliked_items:   Optional[list[str]] = None


class IndexRequest(BaseModel):
    parquet_path: str = "data/processed/train.parquet"


class BatchSimulateRequest(BaseModel):
    records: list[dict]           # list of test records
    apply_adapter: bool = False   # keep False for ROUGE eval


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "NaijaNutri Pro"}


@app.get("/stats")
def stats():
    if not vs:
        raise HTTPException(503, "Vector store not initialised")
    return vs.stats()


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    if not sim:
        raise HTTPException(503, "Simulator not ready")
    try:
        result = sim.simulate(
            user_id=req.user_id,
            item_id=req.item_id,
            item_name=req.item_name,
            item_domain=req.item_domain,
            item_category=req.item_category,
            item_metadata=req.item_metadata,
            apply_nigerian_adapter=req.apply_nigerian_adapter,
        )
        return SimulateResponse(
            user_id=result.user_id,
            item_id=result.item_id,
            item_name=result.item_name,
            predicted_rating=result.predicted_rating,
            review_text=result.review_text,
            raw_review_text=result.raw_review_text,
            reasoning=result.reasoning,
            archetype_name=result.archetype_name,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not rec:
        raise HTTPException(503, "Recommender not ready")
    try:
        result = rec.recommend(
            user_id=req.user_id,
            query=req.query,
            n=req.n,
            domains=req.domains,
            budget_naira=req.budget_naira,
            dietary_restrictions=req.dietary_restrictions,
            calorie_target=req.calorie_target,
            meals_per_day=req.meals_per_day,
            weight_goal=req.weight_goal,
            location=req.location,
            product_interests=req.product_interests,
            book_genres=req.book_genres,
            language_preference=req.language_preference,
            conversation_history=req.conversation_history,
            plan_mode=req.plan_mode,
            exclude_items=req.exclude_items,
        )
        return RecommendResponse(
            user_id=result.user_id,
            query=result.query,
            recommendations=[
                RecommendedItemOut(
                    item_id=r.item_id,
                    item_name=r.item_name,
                    item_domain=r.item_domain,
                    item_category=r.item_category,
                    avg_rating=r.avg_rating,
                    score=r.score,
                    reason=r.reason,
                    budget_ok=r.budget_ok,
                )
                for r in result.recommendations
            ],
            archetype_name=result.archetype_name,
            is_cold_start=result.is_cold_start,
            strategy_used=result.strategy_used,
            response_text=result.response_text,
            meal_plan=result.meal_plan,
            tdee_used=result.tdee_used,
            calories_per_meal=result.calories_per_meal,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/persona")
def build_persona(req: PersonaRequest):
    if not pb:
        raise HTTPException(503, "Persona builder not ready")
    try:
        if req.archetype_key:
            # Explicit cold-start persona from UI inputs
            persona = pb.build_from_explicit(
                user_id=req.user_id,
                archetype_key=req.archetype_key,
                budget_naira=req.budget_naira or 5000,
                dietary_notes=req.dietary_notes,
                liked_categories=req.liked_categories,
                disliked_items=req.disliked_items,
            )
        else:
            persona = pb.build(req.user_id, force_rebuild=req.force_rebuild)

        return {
            "user_id":           persona.user_id,
            "archetype":         persona.archetype.name,
            "budget_tier":       persona.archetype.budget_tier,
            "avg_rating":        persona.avg_rating,
            "category_likes":    persona.category_likes,
            "category_dislikes": persona.category_dislikes,
            "price_sensitivity": persona.price_sensitivity,
            "persona_summary":   persona.persona_summary,
            "is_cold_start":     persona.is_cold_start,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/index/build")
def build_index(req: IndexRequest, background_tasks: BackgroundTasks):
    if not vs:
        raise HTTPException(503, "Vector store not ready")
    if not os.path.exists(req.parquet_path):
        raise HTTPException(400, f"File not found: {req.parquet_path}")
    background_tasks.add_task(build_index_from_parquet, req.parquet_path, vs)
    return {"status": "indexing started", "path": req.parquet_path}


@app.post("/evaluate/batch-simulate")
def batch_simulate(req: BatchSimulateRequest):
    """Run batch simulation for evaluation (used by evaluation scripts)."""
    if not sim:
        raise HTTPException(503, "Simulator not ready")
    results = simulate_batch(sim, req.records, apply_adapter=req.apply_adapter)
    return {"results": results, "count": len(results)}
