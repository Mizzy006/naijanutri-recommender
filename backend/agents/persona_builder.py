"""
backend/agents/persona_builder.py
==================================
Builds a structured UserPersona from a user's review history.
The persona is used by:
  - Task A (review simulator) to calibrate tone + rating bias
  - Task B (recommender)     to understand preferences + constraints

Persona fields
--------------
user_id             str
archetype           Archetype
avg_rating          float
rating_std          float
domain_preferences  dict[domain -> count]
category_likes      list[str]      # categories rated ≥ 4.0
category_dislikes   list[str]      # categories rated < 3.0
liked_items         list[item_id]
disliked_items      list[item_id]
price_sensitivity   "low" | "mid" | "high"
review_length_avg   int            # chars
vocabulary_richness float          # unique words / total words
persona_summary     str            # plain-text summary for LLM context
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from backend.core import llm_client
from backend.core.vector_store import VectorStore
from backend.nigerian_layer.archetypes import Archetype, infer_archetype


# ─── UserPersona dataclass ────────────────────────────────────────────────────

@dataclass
class UserPersona:
    user_id:             str
    archetype:           Archetype
    avg_rating:          float
    rating_std:          float
    domain_preferences:  dict          # {"yelp": 12, "amazon": 3, ...}
    category_likes:      list[str]
    category_dislikes:   list[str]
    liked_items:         list[str]
    disliked_items:      list[str]
    price_sensitivity:   str
    review_length_avg:   int
    vocabulary_richness: float
    persona_summary:     str
    is_cold_start:       bool = False  # True when < 5 reviews available


# ─── Builder ──────────────────────────────────────────────────────────────────

class PersonaBuilder:
    """
    Constructs and caches UserPersona objects.
    Personas are stored in ChromaDB for fast retrieval.
    """

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, user_id: str, force_rebuild: bool = False) -> UserPersona:
        """
        Build or retrieve a persona for `user_id`.
        Pulls the user's review history from ChromaDB.
        """
        # Try cache first
        if not force_rebuild:
            cached = self.vs.get_user_persona(user_id)
            if cached:
                return _deserialise_persona(cached)

        # Fetch review history
        reviews = self.vs.get_user_reviews(user_id, limit=100)

        if len(reviews) < 5:
            return self._build_cold_start(user_id)

        return self._build_from_reviews(user_id, reviews)

    def build_from_explicit(
        self,
        user_id:          str,
        archetype_key:    str,
        budget_naira:     int,
        dietary_notes:    str = "",
        liked_categories: list[str] | None = None,
        disliked_items:   list[str] | None = None,
    ) -> UserPersona:
        """
        Build a persona from explicit user-provided context (e.g. from the UI).
        Used for cold-start in Task B.
        """
        from backend.nigerian_layer.archetypes import get_archetype, ARCHETYPES
        archetype = get_archetype(archetype_key)

        summary = (
            f"User ID: {user_id}. "
            f"Archetype: {archetype.name}. "
            f"Budget: ₦{budget_naira:,}. "
            f"Dietary notes: {dietary_notes or 'none'}. "
            f"Likes: {', '.join(liked_categories or [])}. "
            f"Avoids: {', '.join(disliked_items or [])}."
        )

        persona = UserPersona(
            user_id=user_id,
            archetype=archetype,
            avg_rating=3.5,
            rating_std=0.8,
            domain_preferences={},
            category_likes=liked_categories or [],
            category_dislikes=[],
            liked_items=[],
            disliked_items=disliked_items or [],
            price_sensitivity=archetype.budget_tier,
            review_length_avg=100,
            vocabulary_richness=0.5,
            persona_summary=summary,
            is_cold_start=True,
        )
        self._cache_persona(persona)
        return persona

    # ── Internal builders ─────────────────────────────────────────────────────

    def _build_from_reviews(
        self,
        user_id: str,
        reviews: list[dict],
    ) -> UserPersona:
        ratings = [float(r.get("rating", 3.0)) for r in reviews]
        avg_rating  = statistics.mean(ratings)
        rating_std  = statistics.stdev(ratings) if len(ratings) > 1 else 0.0

        # Domain distribution
        domain_prefs: dict[str, int] = {}
        for r in reviews:
            d = r.get("item_domain", "unknown")
            domain_prefs[d] = domain_prefs.get(d, 0) + 1

        # Category likes / dislikes
        cat_ratings: dict[str, list[float]] = {}
        for r in reviews:
            cat = r.get("item_category", "Unknown")
            cat_ratings.setdefault(cat, []).append(float(r.get("rating", 3.0)))
        cat_avgs = {cat: statistics.mean(rs) for cat, rs in cat_ratings.items()}
        category_likes    = [c for c, a in cat_avgs.items() if a >= 4.0]
        category_dislikes = [c for c, a in cat_avgs.items() if a < 3.0]

        # Liked / disliked items
        liked_items    = [r["item_id"] for r in reviews if float(r.get("rating", 0)) >= 4.0][:20]
        disliked_items = [r["item_id"] for r in reviews if float(r.get("rating", 5)) < 3.0][:20]

        # Review text metrics
        texts         = [r.get("document", "") for r in reviews if r.get("document")]
        avg_len       = int(statistics.mean([len(t) for t in texts])) if texts else 100
        all_words     = " ".join(texts).lower().split()
        vocab_rich    = len(set(all_words)) / max(len(all_words), 1)

        # Archetype inference
        archetype = infer_archetype(
            avg_rating=avg_rating,
            review_history=[r.get("document", "") for r in reviews],
        )

        # Price sensitivity from budget tier
        price_map = {"low": "low", "mid": "mid", "high": "high"}
        price_sens = price_map.get(archetype.budget_tier, "mid")

        # LLM-generated summary
        summary = self._generate_summary(user_id, reviews, archetype, avg_rating)

        persona = UserPersona(
            user_id=user_id,
            archetype=archetype,
            avg_rating=round(avg_rating, 2),
            rating_std=round(rating_std, 2),
            domain_preferences=domain_prefs,
            category_likes=category_likes[:10],
            category_dislikes=category_dislikes[:10],
            liked_items=liked_items,
            disliked_items=disliked_items,
            price_sensitivity=price_sens,
            review_length_avg=avg_len,
            vocabulary_richness=round(vocab_rich, 3),
            persona_summary=summary,
            is_cold_start=False,
        )
        self._cache_persona(persona)
        return persona

    def _build_cold_start(self, user_id: str) -> UserPersona:
        from backend.nigerian_layer.archetypes import ARCHETYPES
        archetype = ARCHETYPES["working_class"]
        summary = (
            f"New user {user_id} with no review history. "
            "Using default Nigerian working-class profile. "
            "Recommendations will be popularity-weighted."
        )
        return UserPersona(
            user_id=user_id,
            archetype=archetype,
            avg_rating=3.5,
            rating_std=0.8,
            domain_preferences={},
            category_likes=[],
            category_dislikes=[],
            liked_items=[],
            disliked_items=[],
            price_sensitivity="mid",
            review_length_avg=100,
            vocabulary_richness=0.5,
            persona_summary=summary,
            is_cold_start=True,
        )

    def _generate_summary(
        self,
        user_id:    str,
        reviews:    list[dict],
        archetype:  Archetype,
        avg_rating: float,
    ) -> str:
        sample = reviews[:5]
        review_snippets = "\n".join(
            f"- [{r.get('item_domain','?')}] {r.get('document','')[:120]} (rated {r.get('rating','?')})"
            for r in sample
        )
        prompt = f"""
Analyse this Nigerian user's review history and write a 2-3 sentence persona summary.
Focus on: their taste preferences, quality/price expectations, and communication style.
Be specific — mention actual patterns you observe.

User archetype detected: {archetype.name}
Average rating given: {avg_rating:.1f}/5.0

Recent reviews:
{review_snippets}

Persona summary (2-3 sentences, plain text):""".strip()

        try:
            return llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                system="You are a user behaviour analyst. Be concise and specific.",
                temperature=0.3,
                max_tokens=150,
            ).strip()
        except Exception:
            return (
                f"User {user_id} ({archetype.name}) with avg rating "
                f"{avg_rating:.1f}. "
                f"Reviews across {len(reviews)} items."
            )

    def _cache_persona(self, persona: UserPersona) -> None:
        self.vs.upsert_user_persona(
            user_id=persona.user_id,
            persona_text=persona.persona_summary,
            meta={
                "archetype":          persona.archetype.name,
                "budget_tier":        persona.archetype.budget_tier,
                "avg_rating":         persona.avg_rating,
                "price_sensitivity":  persona.price_sensitivity,
                "is_cold_start":      persona.is_cold_start,
                "category_likes":     ",".join(persona.category_likes),
                "category_dislikes":  ",".join(persona.category_dislikes),
            },
        )


# ─── Deserialiser ─────────────────────────────────────────────────────────────

def _deserialise_persona(cached: dict) -> UserPersona:
    from backend.nigerian_layer.archetypes import get_archetype, ARCHETYPES
    arch_name  = cached.get("archetype", "working_class")
    arch_key   = next(
        (k for k, a in ARCHETYPES.items() if a.name == arch_name),
        "working_class"
    )
    archetype = get_archetype(arch_key)

    return UserPersona(
        user_id=cached.get("id", ""),
        archetype=archetype,
        avg_rating=float(cached.get("avg_rating", 3.5)),
        rating_std=0.8,
        domain_preferences={},
        category_likes=_split(cached.get("category_likes", "")),
        category_dislikes=_split(cached.get("category_dislikes", "")),
        liked_items=[],
        disliked_items=[],
        price_sensitivity=cached.get("price_sensitivity", "mid"),
        review_length_avg=100,
        vocabulary_richness=0.5,
        persona_summary=cached.get("persona_text", ""),
        is_cold_start=bool(cached.get("is_cold_start", False)),
    )


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]
