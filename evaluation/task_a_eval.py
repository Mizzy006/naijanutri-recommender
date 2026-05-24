"""
evaluation/task_a_eval.py
==========================
Evaluates Task A (User Modeling) on the test set.

Metrics:
  - Rating Accuracy  : RMSE, MAE
  - Review Quality   : ROUGE-1, ROUGE-2, ROUGE-L (F1)
  - Semantic Quality : BERTScore (F1, using roberta-large)

Usage:
    python evaluation/task_a_eval.py \
        --test   data/processed/test.parquet \
        --api    http://localhost:8000 \
        --limit  500 \
        --out    evaluation/results_task_a.json
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
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from tqdm import tqdm


# ─── Scoring functions ────────────────────────────────────────────────────────

def rmse(true: list[float], pred: list[float]) -> float:
    assert len(true) == len(pred)
    return math.sqrt(sum((t - p) ** 2 for t, p in zip(true, pred)) / len(true))


def mae(true: list[float], pred: list[float]) -> float:
    return sum(abs(t - p) for t, p in zip(true, pred)) / len(true)


def rouge_scores(references: list[str], hypotheses: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_f, r2_f, rl_f = [], [], []
    for ref, hyp in zip(references, hypotheses):
        if not hyp.strip():
            hyp = "no review generated"
        s = scorer.score(ref, hyp)
        r1_f.append(s["rouge1"].fmeasure)
        r2_f.append(s["rouge2"].fmeasure)
        rl_f.append(s["rougeL"].fmeasure)
    return {
        "rouge1_f1": round(statistics.mean(r1_f), 4),
        "rouge2_f1": round(statistics.mean(r2_f), 4),
        "rougeL_f1": round(statistics.mean(rl_f), 4),
    }


def bert_scores(references: list[str], hypotheses: list[str]) -> dict:
    clean_hyps = [h if h.strip() else "no review" for h in hypotheses]
    P, R, F1 = bert_score_fn(
        clean_hyps,
        references,
        lang="en",
        model_type="roberta-large",
        verbose=False,
    )
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall":    round(R.mean().item(), 4),
        "bertscore_f1":        round(F1.mean().item(), 4),
    }


# ─── API caller ───────────────────────────────────────────────────────────────

def call_simulate_api(api_url: str, records: list[dict]) -> list[dict]:
    """
    Call /simulate individually per record to avoid batch timeout.
    Adds a small delay between calls to respect Groq rate limits.
    """
    results = []
    with httpx.Client(timeout=60) as client:
        for rec in tqdm(records, desc="Simulating reviews"):
            try:
                resp = client.post(
                    f"{api_url}/simulate",
                    json={
                        "user_id":                rec["user_id"],
                        "item_id":                rec["item_id"],
                        "item_name":              rec.get("item_name", rec["item_id"]),
                        "item_domain":            rec.get("item_domain", "yelp"),
                        "item_category":          rec.get("item_category", "General"),
                        "apply_nigerian_adapter": False,   # raw text for ROUGE eval
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results.append({
                    "user_id":          rec["user_id"],
                    "item_id":          rec["item_id"],
                    "true_rating":      float(rec.get("true_rating", 3.0)),
                    "predicted_rating": data["predicted_rating"],
                    "true_text":        rec.get("true_review_text", ""),
                    "predicted_text":   data["raw_review_text"],
                    "adapted_text":     data["review_text"],
                    "archetype":        data["archetype_name"],
                })
                time.sleep(0.3)   # avoid Groq rate limit
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  default="data/processed/test.parquet")
    parser.add_argument("--api",   default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out",   default="evaluation/results_task_a.json")
    args = parser.parse_args()

    print(f"Loading test set from {args.test}…")
    test_df = pd.read_parquet(args.test).head(args.limit)
    print(f"Test records: {len(test_df):,}")

    records = [
        {
            "user_id":          row["user_id"],
            "item_id":          row["item_id"],
            "item_name":        row.get("item_name", row["item_id"]),
            "item_domain":      row.get("item_domain", "yelp"),
            "item_category":    row.get("item_category", "General"),
            "true_rating":      float(row["rating"]),
            "true_review_text": row["review_text"],
        }
        for _, row in test_df.iterrows()
    ]

    print(f"Calling API at {args.api} for {len(records)} records…")
    start = time.time()
    results = call_simulate_api(args.api, records)
    elapsed = time.time() - start
    print(f"API calls complete in {elapsed:.1f}s")

    # ── Filter valid results ──────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    print(f"Valid results: {len(valid)} / {len(results)}")

    true_ratings = [r["true_rating"]      for r in valid]
    pred_ratings = [r["predicted_rating"] for r in valid]
    true_texts   = [r["true_text"]        for r in valid]
    pred_texts   = [r["predicted_text"]   for r in valid]

    # ── Compute metrics ───────────────────────────────────────────────────────
    rating_metrics = {
        "rmse": round(rmse(true_ratings, pred_ratings), 4),
        "mae":  round(mae(true_ratings,  pred_ratings), 4),
    }
    print(f"Rating metrics: {rating_metrics}")

    text_metrics_rouge = rouge_scores(true_texts, pred_texts)
    print(f"ROUGE:          {text_metrics_rouge}")

    print("Computing BERTScore (this may take a minute)…")
    text_metrics_bert = bert_scores(true_texts, pred_texts)
    print(f"BERTScore:      {text_metrics_bert}")

    # ── Archetype breakdown ───────────────────────────────────────────────────
    arch_rmse: dict[str, list] = {}
    for r in valid:
        arch = r.get("archetype", "unknown")
        arch_rmse.setdefault(arch, [])
        arch_rmse[arch].append((r["true_rating"], r["predicted_rating"]))
    arch_breakdown = {
        arch: round(rmse([x[0] for x in pairs], [x[1] for x in pairs]), 4)
        for arch, pairs in arch_rmse.items()
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "n_evaluated":       len(valid),
        "n_errors":          len(results) - len(valid),
        "elapsed_seconds":   round(elapsed, 1),
        "rating_metrics":    rating_metrics,
        "rouge_metrics":     text_metrics_rouge,
        "bertscore_metrics": text_metrics_bert,
        "archetype_rmse":    arch_breakdown,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
