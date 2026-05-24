"""
backend/core/vector_store.py
============================
ChromaDB wrapper with three collections:

  items    — one doc per unique item (name + category + domain + avg_rating)
  reviews  — one doc per review (text + full metadata)
  users    — one doc per user persona summary (built lazily)

All collections share the same embedding function (all-MiniLM-L6-v2) so
semantic search works across domains seamlessly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# ─── Constants ────────────────────────────────────────────────────────────────

EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_ITEMS    = "naijanutri_items"
COLLECTION_REVIEWS  = "naijanutri_reviews"
COLLECTION_USERS    = "naijanutri_users"

_CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
_CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))


# ─── Client singleton ─────────────────────────────────────────────────────────

def _get_client():
    # Use PersistentClient so it runs locally inside app
    return chromadb.PersistentClient(
        path="./chroma_data",
        settings=Settings(anonymized_telemetry=False)
    )


def _get_ef() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


# ─── VectorStore class ────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around ChromaDB that exposes domain-aware add/query methods.
    """

    def __init__(self):
        self._client = _get_client()
        self._ef     = _get_ef()
        self._items   = self._client.get_or_create_collection(
            name=COLLECTION_ITEMS,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._reviews = self._client.get_or_create_collection(
            name=COLLECTION_REVIEWS,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._users   = self._client.get_or_create_collection(
            name=COLLECTION_USERS,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Items ────────────────────────────────────────────────────────────────

    def upsert_item(
        self,
        item_id:       str,
        item_name:     str,
        item_domain:   str,
        item_category: str,
        avg_rating:    float,
        city:          str = "",
        extra_meta:    dict | None = None,
    ) -> None:
        doc = f"{item_name}. Category: {item_category}. Domain: {item_domain}."
        if city:
            doc += f" Location: {city}."
        meta: dict[str, Any] = {
            "item_name":     item_name,
            "item_domain":   item_domain,
            "item_category": item_category,
            "avg_rating":    round(avg_rating, 2),
            "city":          city,
        }
        if extra_meta:
            # ChromaDB metadata values must be str/int/float/bool
            for k, v in extra_meta.items():
                meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)

        self._items.upsert(
            ids=[item_id],
            documents=[doc],
            metadatas=[meta],
        )

    def query_items(
        self,
        query_text:   str,
        n_results:    int = 20,
        where:        dict | None = None,
    ) -> list[dict]:
        """
        Semantic search over items.
        `where` supports ChromaDB metadata filters, e.g.
            {"item_domain": "yelp"}
            {"$and": [{"item_domain": "yelp"}, {"avg_rating": {"$gte": 4.0}}]}
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results":   n_results,
            "include":     ["metadatas", "distances", "documents"],
        }
        if where:
            kwargs["where"] = where

        res = self._items.query(**kwargs)
        return _unpack_query(res)

    def get_item(self, item_id: str) -> dict | None:
        res = self._items.get(ids=[item_id], include=["metadatas", "documents"])
        if not res["ids"]:
            return None
        return {"id": item_id, "document": res["documents"][0], **res["metadatas"][0]}

    # ── Reviews ───────────────────────────────────────────────────────────────

    def upsert_review(
        self,
        review_id:     str,
        review_text:   str,
        user_id:       str,
        item_id:       str,
        item_domain:   str,
        item_category: str,
        rating:        float,
        timestamp:     int,
        extra_meta:    dict | None = None,
    ) -> None:
        meta: dict[str, Any] = {
            "user_id":       user_id,
            "item_id":       item_id,
            "item_domain":   item_domain,
            "item_category": item_category,
            "rating":        round(rating, 1),
            "timestamp":     timestamp,
        }
        if extra_meta:
            for k, v in extra_meta.items():
                meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)

        self._reviews.upsert(
            ids=[review_id],
            documents=[review_text],
            metadatas=[meta],
        )

    def get_user_reviews(self, user_id: str, limit: int = 50) -> list[dict]:
        """Fetch all stored reviews for a given user."""
        res = self._reviews.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],
            limit=limit,
        )
        return _unpack_get(res)

    def query_similar_reviews(
        self,
        query_text: str,
        n_results:  int = 10,
        where:      dict | None = None,
    ) -> list[dict]:
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results":   n_results,
            "include":     ["metadatas", "distances", "documents"],
        }
        if where:
            kwargs["where"] = where
        res = self._reviews.query(**kwargs)
        return _unpack_query(res)

    # ── Users ─────────────────────────────────────────────────────────────────

    def upsert_user_persona(
        self,
        user_id:       str,
        persona_text:  str,
        meta:          dict,
    ) -> None:
        clean_meta: dict[str, Any] = {}
        for k, v in meta.items():
            clean_meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        self._users.upsert(
            ids=[user_id],
            documents=[persona_text],
            metadatas=[clean_meta],
        )

    def get_user_persona(self, user_id: str) -> dict | None:
        res = self._users.get(ids=[user_id], include=["metadatas", "documents"])
        if not res["ids"]:
            return None
        return {"id": user_id, "persona_text": res["documents"][0], **res["metadatas"][0]}

    def query_similar_users(self, persona_text: str, n_results: int = 5) -> list[dict]:
        res = self._users.query(
            query_texts=[persona_text],
            n_results=n_results,
            include=["metadatas", "distances", "documents"],
        )
        return _unpack_query(res)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "items":   self._items.count(),
            "reviews": self._reviews.count(),
            "users":   self._users.count(),
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _unpack_query(res: dict) -> list[dict]:
    """Flatten Chroma's batched query result into a list of records."""
    records = []
    ids       = res.get("ids",       [[]])[0]
    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    for i, doc_id in enumerate(ids):
        records.append({
            "id":       doc_id,
            "document": documents[i] if i < len(documents) else "",
            "distance": round(distances[i], 4) if i < len(distances) else 1.0,
            **(metadatas[i] if i < len(metadatas) else {}),
        })
    return records


def _unpack_get(res: dict) -> list[dict]:
    """Flatten Chroma's .get() result into a list of records."""
    records = []
    ids       = res.get("ids",       [])
    documents = res.get("documents", [])
    metadatas = res.get("metadatas", [])
    for i, doc_id in enumerate(ids):
        records.append({
            "id":       doc_id,
            "document": documents[i] if i < len(documents) else "",
            **(metadatas[i] if i < len(metadatas) else {}),
        })
    return records


# ─── Index builder (called once at startup) ───────────────────────────────────

def build_index_from_parquet(parquet_path: str, vs: VectorStore, batch_size: int = 512) -> None:
    """
    Load the preprocessed Parquet file and index all items + reviews into
    ChromaDB. Idempotent — uses upsert.
    """
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_parquet(parquet_path)
    print(f"Indexing {len(df):,} records from {parquet_path}")

    # ── Index items (unique) ─────────────────────────────────────────────────
    # Add city column if missing
    if "city" not in df.columns:
        df["city"] = ""
    items_df = (
        df.groupby("item_id")
          .agg(
              item_name=("item_name",         "first"),
              item_domain=("item_domain",     "first"),
              item_category=("item_category", "first"),
              avg_rating=("rating",           "mean"),
              city=("city",                   "first"),
          )
          .reset_index()
    )
    print(f"  Upserting {len(items_df):,} items …")
    for _, row in tqdm(items_df.iterrows(), total=len(items_df)):
        vs.upsert_item(
            item_id=row["item_id"],
            item_name=row["item_name"],
            item_domain=row["item_domain"],
            item_category=row["item_category"],
            avg_rating=row["avg_rating"],
            city=str(row.get("city", "") or ""),
        )

    # ── Index reviews in batches ─────────────────────────────────────────────
    print(f"  Upserting {len(df):,} reviews …")
    for start in tqdm(range(0, len(df), batch_size)):
        chunk = df.iloc[start : start + batch_size]
        for _, row in chunk.iterrows():
            meta_raw = row.get("metadata", {})
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            vs.upsert_review(
                review_id=row["review_id"],
                review_text=row["review_text"],
                user_id=row["user_id"],
                item_id=row["item_id"],
                item_domain=row["item_domain"],
                item_category=row["item_category"],
                rating=float(row["rating"]),
                timestamp=int(row.get("timestamp", 0)),
                extra_meta=meta,
            )

    print(f"\nIndex stats: {vs.stats()}")
