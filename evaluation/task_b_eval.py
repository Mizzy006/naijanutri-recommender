"""
evaluation/task_b_eval.py
==========================
Evaluates Task B (Recommendation) on the test set.

Evaluation protocol (leave-one-out):
  - For each test user, their last interaction is the ground truth.
  - We call /recommend and check if the ground-truth item appears in top-K.

Metrics:
  - NDCG@10   : normalised discounted cumulative gain at K=10
  - Hit Rate@10: fraction of users for whom ground truth is in top-10
  - MRR       : mean reciprocal rank
  - Cold-Start score: NDCG@10 restricted to users with < 5 training interactions

Usage:
    python evaluation/task_b_eval.py \
        --test   data/processed/test.parquet \
        --train  data/processed/train.parquet \
        --api    http://localhost:8000 \
        --limit  500 \
        --out    evaluation/results_task_b.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm


# ─── Metric functions ─────────────────────────────────────────────────────────

def dcg(scores: list[float]) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(ranked_item_ids: list[str], relevant_id: str, k: int = 10) -> float:
    top_k = ranked_item_ids[:k]
    if relevant_id not in top_k:
        return 0.0
    rank = top_k.index(relevant_id)
    actual = dcg([1.0 if i == rank else 0.0 for i in range(k)])
    ideal  = dcg([1.0] + [0.0] * (k - 1))
    return actual / ideal if ideal > 0 else 0.0


def hit_at_k(ranked_item_ids: list[str], relevant_id: str, k: int = 10) -> float:
    return 1.0 if relevant_id in ranked_item_ids[:k] else 0.0


def reciprocal_rank(ranked_item_ids: list[str], relevant_id: str) -> float:
    if relevant_id not in ranked_item_ids:
        return 0.0
    return 1.0 / (ranked_item_ids.index(relevant_id) + 1)


# ─── API caller ───────────────────────────────────────────────────────────────

def call_recommend(
    api_url:  str,
    user_id:  str,
    query:    str,
    domain:   str,
    n:        int = 20,
) -> list[str]:
    """Return ordered list of item_ids from the recommender."""
    with httpx.Client(timeout=90) as client:
        resp = client.post(
            f"{api_url}/recommend",
            json={
                "user_id": user_id,
                "query":   query,
                "n":       n,
                "domains": [domain],
            },
        )
        resp.raise_for_status()
        recs = resp.json().get("recommendations", [])
        return [r["item_id"] for r in recs]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  default="data/processed/test.parquet")
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--api",   default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--k",     type=int, default=10)
    parser.add_argument("--out",   default="evaluation/results_task_b.json")
    args = parser.parse_args()

    print(f"Loading datasets…")
    test_df  = pd.read_parquet(args.test).head(args.limit)
    train_df = pd.read_parquet(args.train)

    # Identify cold-start users (< 5 training interactions)
    train_counts = train_df.groupby("user_id")["review_id"].count()
    cold_users   = set(train_counts[train_counts < 5].index)
    print(f"Cold-start users in test: {len([u for u in test_df['user_id'] if u in cold_users])}")

    ndcg_scores:    list[float] = []
    hit_scores:     list[float] = []
    mrr_scores:     list[float] = []
    cold_ndcg:      list[float] = []
    domain_ndcg:    dict[str, list[float]] = {}
    errors = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        user_id    = row["user_id"]
        true_item  = row["item_id"]
        domain     = row.get("item_domain", "yelp")
        category   = row.get("item_category", "")

        # Use category as the query (simulates a realistic user request)
        query = f"recommend a good {category}" if category else "something good"

        try:
            ranked_ids = call_recommend(
                api_url=args.api,
                user_id=user_id,
                query=query,
                domain=domain,
                n=args.k + 10,
            )

            n_score  = ndcg_at_k(ranked_ids, true_item, args.k)
            h_score  = hit_at_k(ranked_ids,  true_item, args.k)
            rr_score = reciprocal_rank(ranked_ids, true_item)

            ndcg_scores.append(n_score)
            hit_scores.append(h_score)
            mrr_scores.append(rr_score)

            domain_ndcg.setdefault(domain, []).append(n_score)

            if user_id in cold_users:
                cold_ndcg.append(n_score)

            # Small delay to avoid rate limiting
            time.sleep(0.05)

        except Exception as e:
            errors += 1
            ndcg_scores.append(0.0)
            hit_scores.append(0.0)
            mrr_scores.append(0.0)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    output = {
        "n_evaluated":    len(ndcg_scores),
        "n_errors":       errors,
        f"ndcg@{args.k}": round(statistics.mean(ndcg_scores), 4),
        f"hit@{args.k}":  round(statistics.mean(hit_scores),  4),
        "mrr":            round(statistics.mean(mrr_scores),  4),
        "cold_start_ndcg": round(statistics.mean(cold_ndcg), 4) if cold_ndcg else None,
        "n_cold_start":   len(cold_ndcg),
        "domain_ndcg": {
            d: round(statistics.mean(scores), 4)
            for d, scores in domain_ndcg.items()
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
