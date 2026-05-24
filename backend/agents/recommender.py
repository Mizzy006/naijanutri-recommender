"""
backend/agents/recommender.py
==============================
Task B: Recommendation

Each domain (food / products / books) has:
  - Its own LLM context builder  → only relevant info reaches the LLM
  - Its own reranker system prompt
  - Its own narrative system prompt
  - Its own budget

Food:     dietary restrictions + TDEE + weight goal + location
Products: archetype + product budget + interests — NO health context
Books:    archetype + book budget + genres — NO health context
"""

from __future__ import annotations
from dataclasses import dataclass

from backend.agents.persona_builder import PersonaBuilder, UserPersona
from backend.core import llm_client
from backend.core.vector_store import VectorStore


# ─── Dietary keyword map ──────────────────────────────────────────────────────

DIETARY_FORBIDDEN: dict[str, list[str]] = {
    "vegetarian": [
        "chicken", "beef", "pork", "fish", "meat", "suya", "turkey",
        "lamb", "goat", "shrimp", "prawn", "seafood", "bacon", "ham",
        "peppered snail", "catfish", "tilapia", "oxtail", "kilimanjaro",
        "republic", "tastee", "barcelos",
    ],
    "vegan": [
        "chicken", "beef", "pork", "fish", "meat", "suya", "turkey",
        "lamb", "goat", "shrimp", "prawn", "seafood", "bacon", "ham",
        "milk", "cheese", "egg", "butter", "cream", "yogurt", "dairy",
        "cold stone",
    ],
    "no pork":    ["pork", "pig", "bacon", "ham", "lard"],
    "halal":      ["pork", "pig", "bacon", "ham", "alcohol", "wine", "beer"],
    "no seafood": ["fish", "seafood", "shrimp", "prawn", "lobster", "crab", "catfish"],
    "gluten free":["bread", "pasta", "wheat", "flour", "cake", "biscuit", "noodle"],
    "no dairy":   ["milk", "cheese", "butter", "cream", "yogurt", "dairy"],
    "high protein": [],
    "low carb":     [],
    "low calorie":  [],
}


# ─── Domain-specific system prompts ──────────────────────────────────────────

_RERANK_FOOD = """You are a food and nutrition recommendation engine for Nigerian users.
Rank candidates by: dietary compliance FIRST, then taste fit, calorie alignment,
budget, location proximity, and cultural relevance.
NEVER recommend items that violate stated dietary restrictions.
Output ONLY valid JSON."""

_RERANK_PRODUCTS = """You are a product recommendation engine for Nigerian consumers.
Rank candidates by: relevance to the query, value for money, specifications,
and availability in Nigeria. Budget is a hard constraint.
Do NOT mention food, diet, nutrition, or health in your reasoning.
Output ONLY valid JSON."""

_RERANK_BOOKS = """You are a book recommendation engine for Nigerian readers.
Rank candidates by: genre fit, relevance to the query, author quality, and
how well the book matches the user's reading interests and archetype.
Do NOT mention food, diet, nutrition, or health in your reasoning.
Output ONLY valid JSON."""

_NARRATIVE_FOOD_EN = """You are NaijaNutri, a Nigerian AI food and nutrition assistant.
Respond in clear, warm Standard Nigerian English. For each dish:
- Say whether it is a vendor/street food purchase or a home-cooked meal
- Mention the estimated cost in Naira and prep/wait time if relevant
- Explain how it fits their dietary needs and calorie goals
IMPORTANT: Only reference dishes from the provided list. Never invent dishes."""

_NARRATIVE_FOOD_PID = """You are NaijaNutri, a Nigerian AI food assistant wey sabi wetin e good to chop.
Respond in authentic Nigerian Pidgin English. For each dish:
- Tell them if na vendor food dem fit buy quick or if dem go cook am at home
- Mention the price in Naira e.g. "e dey around 1500 naira"
- Use expressions like: "e choke", "abeg", "e dey sweet", "no dulling", "this one correct"
IMPORTANT: Only mention dishes from the provided list. No forming, no inventing food."""

_NARRATIVE_PRODUCTS_EN = """You are NaijaNutri, a Nigerian AI product advisor.
Respond in clear Standard Nigerian English. Focus on specifications, value for money,
and why each product suits the user's needs and budget in Nigeria.
Do NOT mention food, diet, calories, or nutrition."""

_NARRATIVE_PRODUCTS_PID = """You are NaijaNutri, a Nigerian product advisor wey know all the gadgets.
Respond in authentic Nigerian Pidgin. Focus on specs and value.
Use expressions like "this one strong", "value for money dey here", "e go last long".
Do NOT mention food, diet, or nutrition at all."""

_NARRATIVE_BOOKS_EN = """You are NaijaNutri, a Nigerian AI book recommendation assistant.
Respond in clear Standard Nigerian English. Explain why each book fits the user's
interests, reading level, and what they will gain from it.
Do NOT mention food, diet, calories, or nutrition."""

_NARRATIVE_BOOKS_PID = """You are NaijaNutri, a Nigerian book advisor wey don read plenty.
Respond in authentic Nigerian Pidgin. Be enthusiastic about good books.
Use expressions like "this book e choke", "you go enjoy am", "e go open your eye".
Do NOT mention food, diet, or nutrition at all."""


RERANK_PROMPTS   = {
    "yelp":          _RERANK_FOOD,
    "nigerian_food": _RERANK_FOOD,
    "amazon":        _RERANK_PRODUCTS,
    "goodreads":     _RERANK_BOOKS,
}
NARRATIVE_PROMPTS = {
    ("yelp",          "english"): _NARRATIVE_FOOD_EN,
    ("yelp",          "pidgin"):  _NARRATIVE_FOOD_PID,
    ("nigerian_food", "english"): _NARRATIVE_FOOD_EN,
    ("nigerian_food", "pidgin"):  _NARRATIVE_FOOD_PID,
    ("amazon",        "english"): _NARRATIVE_PRODUCTS_EN,
    ("amazon",        "pidgin"):  _NARRATIVE_PRODUCTS_PID,
    ("goodreads",     "english"): _NARRATIVE_BOOKS_EN,
    ("goodreads",     "pidgin"):  _NARRATIVE_BOOKS_PID,
}


# ─── Output dataclasses ───────────────────────────────────────────────────────

@dataclass
class RecommendedItem:
    item_id:       str
    item_name:     str
    item_domain:   str
    item_category: str
    avg_rating:    float
    score:         float
    reason:        str
    budget_ok:     bool
    calorie_note:  str = ""


@dataclass
class RecommendationResult:
    user_id:           str
    query:             str
    recommendations:   list[RecommendedItem]
    archetype_name:    str
    is_cold_start:     bool
    strategy_used:     str
    meal_plan:         list[dict] | None = None
    response_text:     str = ""
    tdee_used:         int | None = None
    calories_per_meal: int | None = None


# ─── Recommender ──────────────────────────────────────────────────────────────

class Recommender:

    def __init__(self, vector_store: VectorStore, persona_builder: PersonaBuilder):
        self.vs = vector_store
        self.pb = persona_builder

    def recommend(
        self,
        user_id:              str,
        query:                str,
        n:                    int = 10,
        domains:              list[str] | None = None,
        budget_naira:         int | None = None,
        # Food-only params
        dietary_restrictions: list[str] | None = None,
        calorie_target:       int | None = None,
        meals_per_day:        int = 3,
        weight_goal:          str = "maintain",
        location:             str = "",
        # Products-only params
        product_interests:    list[str] | None = None,
        # Books-only params
        book_genres:          list[str] | None = None,
        # Shared
        language_preference:  str = "english",
        conversation_history: list[dict] | None = None,
        plan_mode:            bool = False,
        exclude_items:        list[str] | None = None,
    ) -> RecommendationResult:

        persona      = self.pb.build(user_id)
        budget       = budget_naira or self._infer_budget(persona)
        # nigerian_food is always searched alongside yelp for food queries
        raw_domain = (domains or ["yelp"])[0]
        active_domain = raw_domain
        food_domains = [raw_domain]
        if raw_domain == "yelp":
            food_domains = ["yelp", "nigerian_food"]
        restrictions = dietary_restrictions or []

        # Adjust calories for weight goal (food only)
        adjusted_cal  = self._adjust_for_goal(calorie_target, weight_goal)
        calories_per_meal = (adjusted_cal // meals_per_day) if adjusted_cal else None

        enriched = self._enrich_query(query, conversation_history or [], active_domain, location)

        # Retrieve
        search_domains = food_domains if active_domain == "yelp" else [active_domain]
        candidates = self._retrieve_candidates(enriched, persona, search_domains, min(n * 6, 80), location=location)

        # Hard dietary filter (food only)
        if active_domain == "yelp":
            candidates = self._dietary_hard_filter(candidates, restrictions)

        candidates = self._hard_filter(candidates, persona, extra_exclude=exclude_items or [])
        candidates = self._collaborative_boost(candidates, persona)
        candidates = self._budget_filter(candidates, budget)

        strategy = (
            "content_only"   if persona.is_cold_start else
            "collab_boosted" if len(candidates) > n   else
            "hybrid"
        )

        if len(candidates) < 3:
            candidates = self._retrieve_candidates(enriched, persona, [active_domain], n * 5, location=location)
            if active_domain == "yelp":
                candidates = self._dietary_hard_filter(candidates, restrictions)
            strategy = "relaxed"

        # Build domain-specific LLM context (KEY: no cross-domain leakage)
        llm_context = self._build_llm_context(
            persona=persona,
            domain=active_domain,
            budget=budget,
            dietary_restrictions=restrictions,
            calories_per_meal=calories_per_meal,
            weight_goal=weight_goal,
            location=location,
            product_interests=product_interests or [],
            book_genres=book_genres or [],
        )

        reranked = self._llm_rerank(
            candidates=candidates[:n * 3],
            llm_context=llm_context,
            query=enriched,
            domain=active_domain,
            n=n,
        )

        response_text = self._generate_narrative(
            llm_context=llm_context,
            query=query,
            recommendations=reranked,
            domain=active_domain,
            language_preference=language_preference,
            plan_mode=plan_mode,
        )

        meal_plan = None
        if plan_mode and active_domain == "yelp":
            meal_plan = self._generate_meal_plan(reranked, calories_per_meal)

        return RecommendationResult(
            user_id=user_id,
            query=query,
            recommendations=reranked[:n],
            archetype_name=persona.archetype.name,
            is_cold_start=persona.is_cold_start,
            strategy_used=strategy,
            meal_plan=meal_plan,
            response_text=response_text,
            tdee_used=adjusted_cal,
            calories_per_meal=calories_per_meal,
        )

    # ── Domain-specific LLM context builder ──────────────────────────────────

    def _build_llm_context(
        self,
        persona:              UserPersona,
        domain:               str,
        budget:               int,
        dietary_restrictions: list[str],
        calories_per_meal:    int | None,
        weight_goal:          str,
        location:             str,
        product_interests:    list[str],
        book_genres:          list[str],
    ) -> str:
        """
        Build the LLM context string for the reranker and narrative.
        Critically: Products and Books contexts contain ZERO health/diet information.
        """
        archetype = persona.archetype.name

        if domain in ("yelp", "nigerian_food"):
            goal_desc = {
                "lose":     "wants to lose weight — prefer lower calorie, high-satiety options",
                "gain":     "wants to gain weight — prefer calorie-dense, protein-rich options",
                "maintain": "wants to maintain weight — balance taste and nutrition",
            }.get(weight_goal, "")
            diet_line = (
                f"STRICT dietary restrictions (never violate): {', '.join(dietary_restrictions)}"
                if dietary_restrictions else "No dietary restrictions."
            )
            cal_line = (
                f"Calorie target per meal: ~{calories_per_meal} kcal. User {goal_desc}."
                if calories_per_meal else f"User {goal_desc}." if goal_desc else ""
            )
            loc_line = f"User location: {location}." if location else ""
            return (
                f"User archetype: {archetype}\n"
                f"Food budget: ₦{budget:,}\n"
                f"{diet_line}\n"
                f"{cal_line}\n"
                f"{loc_line}\n"
                f"Dataset: Nigerian dishes — some are street food/vendor purchases, "
                f"some are home-cooked meals. Use prep_time_mins and meal_type metadata "
                f"to match the user's urgency and context."
            ).strip()

        elif domain == "amazon":
            interests_line = (
                f"Product interests: {', '.join(product_interests)}."
                if product_interests else "No specific product interests."
            )
            return (
                f"User archetype: {archetype}\n"
                f"Product budget: ₦{budget:,}\n"
                f"{interests_line}\n"
                "Focus: value for money, specs, Nigerian availability."
            ).strip()

        elif domain == "goodreads":
            genres_line = (
                f"Preferred genres: {', '.join(book_genres)}."
                if book_genres else "No genre preference specified."
            )
            return (
                f"User archetype: {archetype}\n"
                f"Book budget: ₦{budget:,}\n"
                f"{genres_line}\n"
                "Focus: story quality, relevance to user's life, Nigerian/African authors welcome."
            ).strip()

        return f"User archetype: {archetype}. Budget: ₦{budget:,}."

    # ── LLM reranker ─────────────────────────────────────────────────────────

    def _llm_rerank(
        self,
        candidates:  list[dict],
        llm_context: str,
        query:       str,
        domain:      str,
        n:           int,
    ) -> list[RecommendedItem]:

        candidate_text = "\n".join(
            f"{i+1}. {c.get('item_name', c['id'])} "
            f"(category: {c.get('item_category','?')}, avg_rating: {c.get('avg_rating','?')}, budget_ok: {c.get('budget_ok', True)})"
            for i, c in enumerate(candidates[:25])
        )

        # ─── AGENT 1: THE PROPOSER ────────────────────────────────────────────────
        proposer_prompt = f"""
        {llm_context}
        QUERY: "{query}"
        CANDIDATES:
        {candidate_text}
        
        Select the top {n + 5} candidates that best match the query and context. 
        JSON format: {{"proposed": [{{"item_index": <int>, "initial_reason": "<string>"}}]}}
        """.strip()

        proposer_system = "You are the Proposer Agent. Select the most relevant items. Output ONLY valid JSON."
        
        proposal_result = llm_client.structured_chat(
            messages=[{"role": "user", "content": proposer_prompt}],
            system=proposer_system,
            temperature=0.3,
        )
        proposed_items = proposal_result.get("proposed", [])

        # ─── AGENT 2: THE CRITIC ──────────────────────────────────────────────────
        critic_prompt = f"""
        {llm_context}
        
        The Proposer Agent suggested these items for the query: "{query}".
        PROPOSED ITEMS:
        {proposed_items}
        
        CANDIDATE DATA (Reference):
        {candidate_text}
        
        Your job is to AUDIT these proposals. 
        1. Ensure no item violates the budget or dietary constraints.
        2. Ensure the item actually fits the {domain} domain rules.
        3. Drop any items that fail the audit.
        4. Select the final top {n} items from those that pass.

        JSON format:
        {{
          "reasoning_trace": "<Explain what you accepted/rejected and why>",
          "ranked": [
            {{
              "item_index": <1-based int matching CANDIDATE DATA>,
              "score": <0.0-1.0>,
              "reason": "<Why it passed the audit>",
              "calorie_note": "<calorie comment for food only, else empty>"
            }}
          ]
        }}
        """.strip()

        critic_system = "You are the Compliance Critic Agent. Ruthlessly audit recommendations against user constraints. Output ONLY valid JSON."

        critic_result = llm_client.structured_chat(
            messages=[{"role": "user", "content": critic_prompt}],
            system=critic_system,
            temperature=0.1, # Low temperature for strict evaluation
        )

        # Print reasoning trace to backend logs for debugging/paper evidence
        print(f"🕵️ Critic Agent Trace: {critic_result.get('reasoning_trace', 'No trace')}")

        output: list[RecommendedItem] = []
        for entry in critic_result.get("ranked", [])[:n]:
            idx = int(entry.get("item_index", 1)) - 1
            if idx < 0 or idx >= len(candidates):
                continue
            c = candidates[idx]
            output.append(RecommendedItem(
                item_id=c["id"],
                item_name=c.get("item_name", c["id"]),
                item_domain=c.get("item_domain", domain),
                item_category=c.get("item_category", "General"),
                avg_rating=float(c.get("avg_rating", 3.5)),
                score=float(entry.get("score", 0.5)),
                reason=entry.get("reason", "Passed compliance audit."),
                budget_ok=bool(c.get("budget_ok", True)),
                calorie_note=entry.get("calorie_note", "") if domain in ["yelp", "nigerian_food"] else "",
            ))

        # ─── FALLBACK ─────────────────────────────────────────────────────────────
        if not output:
            for c in candidates[:n]:
                output.append(RecommendedItem(
                    item_id=c["id"],
                    item_name=c.get("item_name", c["id"]),
                    item_domain=c.get("item_domain", domain),
                    item_category=c.get("item_category", "General"),
                    avg_rating=float(c.get("avg_rating", 3.5)),
                    score=max(0.0, 1.0 - float(c.get("distance", 0.5))),
                    reason="Matched to your query via fallback.",
                    budget_ok=bool(c.get("budget_ok", True)),
                ))
        return output
    
    # ── Narrative ─────────────────────────────────────────────────────────────

    def _generate_narrative(
        self,
        llm_context:         str,
        query:               str,
        recommendations:     list[RecommendedItem],
        domain:              str,
        language_preference: str,
        plan_mode:           bool,
    ) -> str:
        if not recommendations:
            msgs = {
                "english": "Nothing matched your criteria. Try adjusting your filters.",
                "pidgin":  "Abeg nothing match. Try change the filter small.",
            }
            return msgs.get(language_preference, msgs["english"])

        items_text = "\n".join(
            f"  {i+1}. {r.item_name} ({r.item_category}) — {r.reason}"
            + (f" [{r.calorie_note}]" if r.calorie_note else "")
            for i, r in enumerate(recommendations[:5])
        )

        domain_instruction = {
            "yelp":      "Focus on taste, nutrition, and dietary fit.",
            "amazon":    "Focus on specs, value for money, and usefulness. No food or diet mention.",
            "goodreads": "Focus on story quality and why this book suits them. No food or diet mention.",
        }.get(domain, "")

        plan_note = "Briefly suggest how these form a good weekly plan." if plan_mode and domain == "yelp" else ""

        prompt = f"""
{llm_context}

Query: "{query}"
{domain_instruction}
{plan_note}

Recommendations:
{items_text}

Write 2-3 warm, specific sentences presenting these to the user.
IMPORTANT: Only mention items from the list above. Do not invent, hallucinate,
or suggest any products, restaurants, or books not listed above.
""".strip()

        system = NARRATIVE_PROMPTS.get((domain, language_preference), _NARRATIVE_FOOD_EN)

        return llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            temperature=0.8,
            max_tokens=200,
        ).strip()

    # ── Dietary hard filter (food only) ───────────────────────────────────────

    def _dietary_hard_filter(self, candidates: list[dict], restrictions: list[str]) -> list[dict]:
        if not restrictions:
            return candidates

        forbidden: set[str] = set()
        require_high_protein = False
        require_low_glycemic = False

        for r in restrictions:
            r_lower = r.lower().strip()
            for key, keywords in DIETARY_FORBIDDEN.items():
                if key in r_lower or r_lower in key:
                    forbidden.update(keywords)
            if "high protein" in r_lower:
                require_high_protein = True
            if "low carb" in r_lower or "low glycemic" in r_lower:
                require_low_glycemic = True

        def allowed(c: dict) -> bool:
            name_cat  = f"{c.get('item_name','').lower()} {c.get('item_category','').lower()}"
            allergens = str(c.get("allergens", "")).lower()
            combined  = f"{name_cat} {allergens}"
            if forbidden and any(word in combined for word in forbidden):
                return False
            if require_high_protein:
                protein = str(c.get("protein_level", "")).lower()
                if protein and protein not in ("", "none", "nan") and protein == "low":
                    return False
            if require_low_glycemic:
                glycemic = str(c.get("glycemic_index", "")).lower()
                if glycemic and glycemic not in ("", "none", "nan") and glycemic == "high":
                    return False
            return True

        filtered = [c for c in candidates if allowed(c)]
        return filtered if filtered else candidates


    # ── Candidate retrieval ───────────────────────────────────────────────────

    def _retrieve_candidates(
        self, query: str, persona: UserPersona, domains: list[str],
        n_candidates: int, location: str = "",
    ) -> list[dict]:
        all_candidates: list[dict] = []
        per_domain  = max(3, n_candidates // len(domains))
        taste_query = f"{query} {chr(39).join(persona.category_likes[:3])}"

        for domain in domains:
            # For Yelp + location: city-filtered first, fall back to unfiltered
            if location and domain == "yelp":
                try:
                    loc_results = self.vs.query_items(
                        query_text=taste_query,
                        n_results=per_domain,
                        where={"$and": [
                            {"item_domain": {"$eq": domain}},
                            {"city": {"$eq": location}},
                        ]},
                    )
                    if len(loc_results) >= 3:
                        all_candidates.extend(loc_results)
                        continue
                    all_candidates.extend(loc_results)
                    extra = self.vs.query_items(
                        query_text=taste_query,
                        n_results=per_domain,
                        where={"item_domain": domain},
                    )
                    all_candidates.extend(extra)
                    continue
                except Exception:
                    pass

            results = self.vs.query_items(
                query_text=taste_query,
                n_results=per_domain,
                where={"item_domain": domain},
            )
            all_candidates.extend(results)

        seen: set[str] = set()
        unique: list[dict] = []
        for c in all_candidates:
            if c["id"] not in seen:
                seen.add(c["id"])
                unique.append(c)
        return unique

    def _hard_filter(
        self, candidates: list[dict], persona: UserPersona,
        extra_exclude: list[str] | None = None
    ) -> list[dict]:
        seen = set(persona.liked_items + persona.disliked_items + (extra_exclude or []))
        return [c for c in candidates if c["id"] not in seen]

    def _collaborative_boost(self, candidates: list[dict], persona: UserPersona) -> list[dict]:
        if persona.is_cold_start:
            return candidates
        similar = self.vs.query_similar_users(persona.persona_summary, n_results=5)
        boosted: set[str] = set()
        for u in similar:
            reviews = self.vs.get_user_reviews(u["id"], limit=20)
            boosted.update(r["item_id"] for r in reviews if float(r.get("rating", 0)) >= 4.0)
        for c in candidates:
            c["collab_boost"] = 0.15 if c["id"] in boosted else 0.0
        return candidates

    def _budget_filter(self, candidates: list[dict], budget: int) -> list[dict]:
        def passes(c: dict) -> bool:
            # Exact cost filter for Nigerian food dataset
            est_cost = c.get("est_cost_ngn")
            if est_cost and str(est_cost) not in ("", "0", "None"):
                try:
                    return int(float(str(est_cost))) <= budget
                except (ValueError, TypeError):
                    pass
            # Yelp tier fallback
            tier = str(c.get("price", c.get("metadata_price", "")))
            if not tier or tier == "None":
                return True
            return {"1": 1500, "2": 5000, "3": 15000, "4": 100000}.get(tier, 5000) <= budget
        for c in candidates:
            c["budget_ok"] = passes(c)
        filtered = [c for c in candidates if passes(c)]
        return filtered if filtered else candidates

    def _generate_meal_plan(
        self, items: list[RecommendedItem], calories_per_meal: int | None
    ) -> list[dict]:
        food_items = [i for i in items if i.item_domain == "yelp"] or items
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        return [
            {
                "day":          day,
                "item_name":    food_items[i % len(food_items)].item_name,
                "category":     food_items[i % len(food_items)].item_category,
                "reason":       food_items[i % len(food_items)].reason,
                "calorie_note": food_items[i % len(food_items)].calorie_note or (
                    f"~{calories_per_meal} kcal target" if calories_per_meal else ""
                ),
            }
            for i, day in enumerate(days)
        ]

    def _adjust_for_goal(self, tdee: int | None, goal: str) -> int | None:
        if not tdee:
            return None
        if goal == "lose":
            return max(1200, tdee - 500)
        elif goal == "gain":
            return tdee + 300
        return tdee

    def _infer_budget(self, persona: UserPersona) -> int:
        return {"low": 3000, "mid": 12000, "high": 50000}.get(
            persona.price_sensitivity, 10000
        )

    def _enrich_query(
        self, query: str, history: list[dict], domain: str, location: str
    ) -> str:
        loc_hint = f" in {location}" if location and domain == "yelp" else ""
        context  = " ".join(h.get("content", "") for h in history[-2:])
        return f"{context} {query}{loc_hint}".strip()[:500]
