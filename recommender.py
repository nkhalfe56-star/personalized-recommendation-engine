import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Personalized Recommendation Engine", version="1.0")


# ── Collaborative Filtering ────────────────────────────────────────────────
class CollaborativeFilter:
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_index = {}
        self.item_index = {}
        self.reverse_item_index = {}

    def fit(self, interactions_df: pd.DataFrame):
        """interactions_df: columns [user_id, item_id, rating]"""
        users = interactions_df["user_id"].unique()
        items = interactions_df["item_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {it: i for i, it in enumerate(items)}
        self.reverse_item_index = {i: it for it, i in self.item_index.items()}

        rows = interactions_df["user_id"].map(self.user_index)
        cols = interactions_df["item_id"].map(self.item_index)
        vals = interactions_df["rating"].values

        matrix = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_.T
        logger.info(f"Fitted CF model: {len(users)} users, {len(items)} items")

    def recommend(self, user_id: str, n: int = 10, exclude_seen: bool = True,
                   seen_items: Optional[List[str]] = None) -> List[str]:
        if user_id not in self.user_index:
            logger.warning(f"Cold start: user {user_id} not found")
            return []
        uid = self.user_index[user_id]
        scores = self.user_factors[uid] @ self.item_factors.T
        item_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        seen_ids = set()
        if exclude_seen and seen_items:
            seen_ids = {self.item_index.get(i) for i in seen_items if i in self.item_index}

        recs = []
        for item_idx, score in item_scores:
            if item_idx not in seen_ids:
                recs.append(self.reverse_item_index[item_idx])
            if len(recs) >= n:
                break
        return recs


# ── Content-Based Filtering ────────────────────────────────────────────────
class ContentFilter:
    def __init__(self):
        self.item_matrix = None
        self.item_ids = []
        self.item_index = {}

    def fit(self, item_features_df: pd.DataFrame):
        """item_features_df: first column is item_id, rest are numeric features"""
        self.item_ids = item_features_df.iloc[:, 0].tolist()
        self.item_index = {item: i for i, item in enumerate(self.item_ids)}
        self.item_matrix = item_features_df.iloc[:, 1:].values.astype(float)
        # Normalize
        norms = np.linalg.norm(self.item_matrix, axis=1, keepdims=True)
        self.item_matrix = self.item_matrix / (norms + 1e-8)
        logger.info(f"Fitted content-based model: {len(self.item_ids)} items")

    def similar_items(self, item_id: str, n: int = 10) -> List[str]:
        if item_id not in self.item_index:
            return []
        idx = self.item_index[item_id]
        sims = cosine_similarity([self.item_matrix[idx]], self.item_matrix)[0]
        top_indices = np.argsort(sims)[::-1][1:n + 1]  # exclude self
        return [self.item_ids[i] for i in top_indices]


# ── API ───────────────────────────────────────────────────────────────────────
cf_model = CollaborativeFilter(n_components=50)
cb_model = ContentFilter()


class RecommendRequest(BaseModel):
    user_id: str
    n: int = 10
    method: str = "hybrid"  # "cf", "content", "hybrid"
    seen_items: Optional[List[str]] = None
    seed_item: Optional[str] = None


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[str]
    method: str
    count: int


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    recs = []
    if req.method in ("cf", "hybrid"):
        cf_recs = cf_model.recommend(
            req.user_id, n=req.n, seen_items=req.seen_items
        )
        recs.extend(cf_recs)
    if req.method in ("content", "hybrid") and req.seed_item:
        cb_recs = cb_model.similar_items(req.seed_item, n=req.n)
        # Merge uniquely
        for item in cb_recs:
            if item not in recs:
                recs.append(item)
    recs = recs[:req.n]
    return RecommendResponse(
        user_id=req.user_id,
        recommendations=recs,
        method=req.method,
        count=len(recs),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
