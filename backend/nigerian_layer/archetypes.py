"""
backend/nigerian_layer/archetypes.py
=====================================
Nigerian user archetypes used to enrich user personas and adapt
generated language.  Each archetype carries:

  - budget_tier     : economic bracket
  - review_style    : how they write reviews
  - vocab_signals   : lexical markers to detect this archetype in existing reviews
  - slang_pool      : expressions the archetype might naturally use
  - rating_bias     : tendency to rate higher (+) or lower (-)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Archetype:
    name:         str
    budget_tier:  str           # "low" | "mid" | "high"
    naira_range:  tuple[int, int]  # approx monthly spend on reviewed category
    description:  str
    review_style: str
    vocab_signals: list[str]
    slang_pool:    list[str]
    rating_bias:   float        # -0.5 to +0.5 adjustment on top of real preference


# ─── Archetype definitions ────────────────────────────────────────────────────

ARCHETYPES: dict[str, Archetype] = {

    "sapa_student": Archetype(
        name="Student",
        budget_tier="low",
        naira_range=(0, 3_000),
        description=(
            "Broke student surviving on vibes and ₦500 meals. "
            "Price is EVERYTHING. Will walk 20 minutes to save ₦200."
        ),
        review_style=(
            "Short, punchy, price-focused reviews. "
            "Uses Gen Z slang heavily. Rates based on value-for-money. "
            "Complains when quantity shrinks. Celebrates any free extra."
        ),
        vocab_signals=["cheap", "affordable", "broke", "student", "budget", "price"],
        slang_pool=[
            "e don do", "sapa no gree", "my pocket dey cry", "e cheap well well",
            "na vibes", "this one sweet pass my expectation",
            "abeg the quantity don reduce o", "hunger is not beans",
            "I no get money to waste", "value for money dey here",
        ],
        rating_bias=+0.1,  # generous when value is good
    ),

    "hustling_corper": Archetype(
        name="Hustling Corper",
        budget_tier="low",
        naira_range=(3_000, 8_000),
        description=(
            "NYSC member on ₦77k monthly allowance. "
            "Seeks midrange options. Compares everything to home state."
        ),
        review_style=(
            "Medium-length reviews comparing to home region. "
            "Notices authenticity issues. Fairly articulate. "
            "Mentions NYSC context occasionally."
        ),
        vocab_signals=["corper", "allawee", "PPA", "orientation", "nysc"],
        slang_pool=[
            "e nor reach the one wey dey my state",
            "for Lagos this one na win",
            "abeg the portion small",
            "e choke for real",
            "my allawee appreciate this place",
        ],
        rating_bias=0.0,
    ),

    "working_class": Archetype(
        name="Working Class",
        budget_tier="mid",
        naira_range=(8_000, 25_000),
        description=(
            "9-to-5 employee treating themselves. "
            "Values consistency and convenience. Will pay for quality."
        ),
        review_style=(
            "Balanced, moderate-length reviews. "
            "Comments on service speed (busy schedule), consistency, "
            "and whether it's worth coming back. Practical tone."
        ),
        vocab_signals=["office", "lunch", "work", "quick", "consistent", "colleague"],
        slang_pool=[
            "after work stress this one reset me",
            "the service fast, I no wait long",
            "e consistent, I come here often",
            "recommend for busy people",
            "work stress don meet their match",
        ],
        rating_bias=0.0,
    ),

    "tech_bro": Archetype(
        name="Tech Bro",
        budget_tier="mid",
        naira_range=(15_000, 60_000),
        description=(
            "Software engineer, product designer, or startup founder. "
            "Works from cafes. Cares about WiFi, ambiance, and aesthetic."
        ),
        review_style=(
            "Concise, structured reviews (sometimes bullet-like). "
            "Focuses on ambiance, WiFi quality, noise level for remote work. "
            "Drops English and Pidgin English mix (code-switching). "
            "May reference tech culture."
        ),
        vocab_signals=["wifi", "ambiance", "remote", "cowork", "aesthetic", "laptop"],
        slang_pool=[
            "e dey functional", "the vibe dey 10/10",
            "WiFi strong pass my package", "the noise level acceptable for calls",
            "aesthetic on point", "I don deploy from here twice",
            "the electricity stable — respect", "for the culture",
        ],
        rating_bias=+0.2,
    ),

    "omo_landlord": Archetype(
        name="Omo Landlord",
        budget_tier="high",
        naira_range=(60_000, 500_000),
        description=(
            "High-earner with disposable income. "
            "Expects premium quality. Very critical of mediocrity. "
            "Has traveled, so benchmarks against international standards."
        ),
        review_style=(
            "Detailed, critical reviews. "
            "Compares to international equivalents. High standards. "
            "Writes in confident Nigerian English, occasionally Pidgin for emphasis. "
            "Low tolerance for poor service."
        ),
        vocab_signals=["premium", "quality", "standard", "expensive", "disappointing"],
        slang_pool=[
            "e nor reach the standard I expect",
            "for this price, abeg do better",
            "I don see better for abroad",
            "e come correct this time",
            "the service level dey embarrassing",
            "this one worth every kobo",
            "they know their onions",
        ],
        rating_bias=-0.2,  # harder to impress
    ),

    "naija_mama": Archetype(
        name="Naija Mama",
        budget_tier="mid",
        naira_range=(5_000, 30_000),
        description=(
            "Homemaker or market woman. Deep food knowledge. "
            "Compares everything to home-cooked standards. Practical buyer."
        ),
        review_style=(
            "Warm, food-focused reviews. Detailed on spice levels, portion size, "
            "freshness of ingredients. Compares to home cooking. "
            "Writes in confident Pidgin/English mix."
        ),
        vocab_signals=["cook", "spice", "pepper", "fresh", "portion", "recipe", "taste"],
        slang_pool=[
            "e nor sweet like mama own",
            "the pepper na correct level",
            "fresh ingredient dey inside, I fit tell",
            "I go teach them how to season",
            "the portion fit satisfy grown man",
            "e come out correct",
            "the aroma enter my soul",
        ],
        rating_bias=+0.0,
    ),
}


# ─── Archetype inference ──────────────────────────────────────────────────────

def infer_archetype(
    avg_rating:     float,
    review_history: list[str],
    avg_price_tier: str | None = None,
) -> Archetype:
    """
    Infer the most likely archetype from a user's review history using
    keyword matching on vocab_signals.
    Falls back to working_class if no signals are detected.
    """
    combined_text = " ".join(review_history).lower()
    scores: dict[str, int] = {k: 0 for k in ARCHETYPES}

    for key, arch in ARCHETYPES.items():
        for signal in arch.vocab_signals:
            if signal in combined_text:
                scores[key] += 1

    # Budget tier hint if available
    if avg_price_tier:
        tier_map = {
            "1": ["sapa_student", "hustling_corper"],
            "2": ["working_class", "tech_bro"],
            "3": ["tech_bro", "omo_landlord"],
            "4": ["omo_landlord"],
        }
        for key in tier_map.get(str(avg_price_tier), []):
            scores[key] += 2

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return ARCHETYPES["working_class"]
    return ARCHETYPES[best]


def get_archetype(name: str) -> Archetype:
    return ARCHETYPES.get(name, ARCHETYPES["working_class"])


def list_archetypes() -> list[dict]:
    return [
        {
            "key":          k,
            "name":         a.name,
            "budget_tier":  a.budget_tier,
            "naira_range":  a.naira_range,
            "description":  a.description,
        }
        for k, a in ARCHETYPES.items()
    ]
