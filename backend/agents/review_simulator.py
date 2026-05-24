"""
backend/agents/review_simulator.py
====================================
Task A: User Modeling

Given a user persona and an unseen item, this agent:
  1. Retrieves the user's review history from ChromaDB
  2. Finds similar items the user has reviewed (few-shot context)
  3. Predicts a star rating (float, 1.0–5.0) via LLM reasoning
  4. Generates a review text that matches the user's style
  5. Passes the review through the Nigerian Language Adapter

Evaluated on:
  - Rating Accuracy (RMSE)
  - Review Text Quality (ROUGE / BERTScore)
  - Behavioural Fidelity (human eval)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.agents.persona_builder import PersonaBuilder, UserPersona
from backend.core import llm_client
from backend.core.vector_store import VectorStore
from backend.nigerian_layer.archetypes import Archetype
from backend.nigerian_layer.language_adapter import adapt_review, rating_to_naija_phrase


# ─── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class SimulatedReview:
    user_id:          str
    item_id:          str
    item_name:        str
    item_domain:      str
    predicted_rating: float        # 1.0–5.0
    review_text:      str          # Nigerian-adapted review
    raw_review_text:  str          # Pre-adaptation (for ROUGE eval)
    reasoning:        str          # LLM's rating justification
    archetype_name:   str


# ─── System prompts ───────────────────────────────────────────────────────────

_RATING_SYSTEM = """You are a behavioural prediction model.
Given a user's review history and persona, predict how they would rate an item they have never seen.
Think step by step: what patterns in their history suggest they would like or dislike this item?
Output ONLY valid JSON."""

_REVIEW_SYSTEM = """You are a review generation model.
Write an authentic review that this specific user would write, based on their history and personality.
Match their typical review length, vocabulary level, and the aspects they normally comment on.
Write in plain English — a separate system will handle localisation.
Output ONLY the review text, nothing else."""


# ─── ReviewSimulator class ────────────────────────────────────────────────────

class ReviewSimulator:

    def __init__(self, vector_store: VectorStore, persona_builder: PersonaBuilder):
        self.vs = vector_store
        self.pb = persona_builder

    # ── Main entry point ──────────────────────────────────────────────────────

    def simulate(
        self,
        user_id:       str,
        item_id:       str,
        item_name:     str,
        item_domain:   str,
        item_category: str,
        item_metadata: dict | None = None,
        apply_nigerian_adapter: bool = True,
    ) -> SimulatedReview:
        """
        Simulate a review for `item_id` by user `user_id`.
        """
        # 1. Build user persona
        persona = self.pb.build(user_id)

        # 2. Retrieve item info from vector store
        item_info = self.vs.get_item(item_id)
        item_avg_rating = float(item_info.get("avg_rating", 3.5)) if item_info else 3.5

        # 3. Find few-shot examples: similar items this user has reviewed
        few_shot = self._get_few_shot_reviews(
            persona=persona,
            item_name=item_name,
            item_domain=item_domain,
            item_category=item_category,
        )

        # 4. Predict rating
        predicted_rating, reasoning = self._predict_rating(
            persona=persona,
            item_name=item_name,
            item_domain=item_domain,
            item_category=item_category,
            item_avg_rating=item_avg_rating,
            item_metadata=item_metadata or {},
            few_shot=few_shot,
        )

        # 5. Generate review text
        raw_review = self._generate_review_text(
            persona=persona,
            item_name=item_name,
            item_domain=item_domain,
            item_category=item_category,
            predicted_rating=predicted_rating,
            few_shot=few_shot,
        )

        # 6. Apply Nigerian language adapter
        if apply_nigerian_adapter:
            adapted_review = adapt_review(
                original_review=raw_review,
                archetype=persona.archetype,
                item_name=item_name,
                item_domain=item_domain,
                star_rating=predicted_rating,
            )
        else:
            adapted_review = raw_review

        return SimulatedReview(
            user_id=user_id,
            item_id=item_id,
            item_name=item_name,
            item_domain=item_domain,
            predicted_rating=round(predicted_rating, 1),
            review_text=adapted_review,
            raw_review_text=raw_review,
            reasoning=reasoning,
            archetype_name=persona.archetype.name,
        )

    # ── Few-shot retrieval ────────────────────────────────────────────────────

    def _get_few_shot_reviews(
        self,
        persona:       UserPersona,
        item_name:     str,
        item_domain:   str,
        item_category: str,
        n:             int = 4,
    ) -> list[dict]:
        """
        Retrieve the most relevant past reviews from this user as few-shot context.
        Prioritises same domain and same category.
        """
        query = f"{item_name} {item_category}"

        # Try same domain + category first
        results = self.vs.query_similar_reviews(
            query_text=query,
            n_results=n,
            where={
                "$and": [
                    {"user_id": persona.user_id},
                    {"item_domain": item_domain},
                ]
            },
        )

        # Fall back to any domain if sparse
        if len(results) < 2:
            results = self.vs.query_similar_reviews(
                query_text=query,
                n_results=n,
                where={"user_id": persona.user_id},
            )
        return results[:n]

    # ── Rating prediction ─────────────────────────────────────────────────────

    def _predict_rating(
        self,
        persona:          UserPersona,
        item_name:        str,
        item_domain:      str,
        item_category:    str,
        item_avg_rating:  float,
        item_metadata:    dict,
        few_shot:         list[dict],
    ) -> tuple[float, str]:
        """
        Use the LLM to predict a rating (1.0–5.0) with chain-of-thought reasoning.
        Returns (predicted_rating, reasoning_text).
        """
        few_shot_text = "\n".join(
            f"  - '{r.get('document','')[:100]}' → rated {r.get('rating', '?')}/5"
            for r in few_shot
        ) or "  (no history available)"

        prompt = f"""
USER PERSONA:
{persona.persona_summary}
Average rating given: {persona.avg_rating}/5.0 (std: {persona.rating_std})
Categories they love: {', '.join(persona.category_likes) or 'unknown'}
Categories they dislike: {', '.join(persona.category_dislikes) or 'none'}
Price sensitivity: {persona.price_sensitivity}

PAST REVIEWS (few-shot):
{few_shot_text}

TARGET ITEM:
Name: {item_name}
Domain: {item_domain}
Category: {item_category}
Overall avg rating from all users: {item_avg_rating}/5.0
Extra info: {item_metadata}

TASK:
Predict the rating this user would give the target item.
Think through:
  1. Does this item match their taste profile?
  2. How have they rated similar items?
  3. Any red flags (price, category misfit, etc.)?
  4. Apply their personal rating bias (do they rate high or low generally?).

Respond with JSON: {{"rating": <float 1.0-5.0>, "reasoning": "<2-3 sentences>"}}
""".strip()

        result = llm_client.structured_chat(
            messages=[{"role": "user", "content": prompt}],
            system=_RATING_SYSTEM,
            temperature=0.2,
        )

        raw_rating = float(result.get("rating", 3.0))
        # Apply archetype bias
        biased_rating = raw_rating + persona.archetype.rating_bias
        clamped = max(1.0, min(5.0, biased_rating))
        reasoning = result.get("reasoning", "No reasoning provided.")
        return round(clamped, 1), reasoning

    # ── Review generation ─────────────────────────────────────────────────────

    def _generate_review_text(
        self,
        persona:          UserPersona,
        item_name:        str,
        item_domain:      str,
        item_category:    str,
        predicted_rating: float,
        few_shot:         list[dict],
    ) -> str:
        """Generate the base review text (pre-Nigerian adaptation)."""
        style_examples = "\n".join(
            f"  \"{r.get('document','')[:150]}\""
            for r in few_shot[:3]
        ) or "  (no examples)"

        target_len = max(50, min(300, persona.review_length_avg))
        sentiment  = (
            "positive and enthusiastic" if predicted_rating >= 4.0 else
            "mixed, with clear reservations" if predicted_rating >= 2.5 else
            "negative and critical"
        )

        prompt = f"""
USER PROFILE:
- Avg review length: {persona.review_length_avg} characters
- Rating they gave: {predicted_rating}/5.0
- Sentiment expected: {sentiment}

THEIR PAST REVIEW STYLE (mimic this voice and length):
{style_examples}

ITEM TO REVIEW:
"{item_name}" ({item_category}, {item_domain} platform)

Write a {target_len}-character review that this user would write.
Focus on the aspects they typically mention.
Match their vocabulary level (richness score: {persona.vocabulary_richness:.2f}).
Write in plain English — do not add Pidgin yourself.
""".strip()

        return llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=_REVIEW_SYSTEM,
            temperature=0.75,
            max_tokens=300,
        ).strip()


# ─── Batch simulation (for evaluation) ───────────────────────────────────────

def simulate_batch(
    simulator:    ReviewSimulator,
    test_records: list[dict],
    apply_adapter: bool = False,   # Disable for ROUGE eval (compare raw English)
) -> list[dict]:
    """
    Run the simulator over the test set.
    test_records: list of {user_id, item_id, item_name, item_domain, item_category,
                           true_rating, true_review_text}
    Returns list of {true_rating, predicted_rating, true_text, predicted_text, ...}
    """
    results = []
    for rec in test_records:
        try:
            sim = simulator.simulate(
                user_id=rec["user_id"],
                item_id=rec["item_id"],
                item_name=rec.get("item_name", rec["item_id"]),
                item_domain=rec.get("item_domain", "unknown"),
                item_category=rec.get("item_category", "General"),
                apply_nigerian_adapter=apply_adapter,
            )
            results.append({
                "user_id":           rec["user_id"],
                "item_id":           rec["item_id"],
                "true_rating":       float(rec.get("true_rating", 3.0)),
                "predicted_rating":  sim.predicted_rating,
                "true_text":         rec.get("true_review_text", ""),
                "predicted_text":    sim.raw_review_text,
                "adapted_text":      sim.review_text,
                "archetype":         sim.archetype_name,
            })
        except Exception as e:
            results.append({
                "user_id":          rec["user_id"],
                "item_id":          rec["item_id"],
                "true_rating":      float(rec.get("true_rating", 3.0)),
                "predicted_rating": 3.0,
                "true_text":        rec.get("true_review_text", ""),
                "predicted_text":   "",
                "adapted_text":     "",
                "archetype":        "error",
                "error":            str(e),
            })
    return results