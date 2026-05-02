from __future__ import annotations

import os
import re
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class RecommenderConfig:
    numeric_cols: Tuple[str, ...] = (
        "Engine HP", "Engine Cylinders", "highway MPG", "city mpg", "MSRP",
    )
    categorical_cols: Tuple[str, ...] = (
        "Engine Fuel Type", "Transmission Type", "Driven_Wheels",
        "Vehicle Size", "Vehicle Style",
    )
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "MSRP": 2.0,
        "Engine HP": 1.5,
        "Vehicle Style": 1.3,
    })
    metric: str = "cosine"
    n_neighbors: int = 25
    weights_dir: str = "./weights"
    artifact_name: str = "car_recommender.joblib"


class CarRecommender:

    DISPLAY_COLS = ("car_name", "MSRP", "Engine HP", "Vehicle Style")

    def __init__(self, config: Optional[RecommenderConfig] = None):
        self.cfg = config or RecommenderConfig()
        self.pipeline: Optional[Pipeline] = None
        self.nn: Optional[NearestNeighbors] = None
        self.df: Optional[pd.DataFrame] = None
        self._weights_vec: Optional[np.ndarray] = None
        self._feat_matrix: Optional[np.ndarray] = None

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).lower().strip())

    def _required_cols(self) -> List[str]:
        return list(self.cfg.numeric_cols) + list(self.cfg.categorical_cols)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=self._required_cols()).reset_index(drop=True)
        df["car_name"] = (df["Make"].astype(str) + " "
                          + df["Model"].astype(str) + " "
                          + df["Year"].astype(str))
        df["_key"] = df["car_name"].map(self._norm)
        return df

    def _build_pipeline(self) -> Pipeline:
        pre = ColumnTransformer([
            ("num", StandardScaler(), list(self.cfg.numeric_cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(self.cfg.categorical_cols)),
        ])
        return Pipeline([("pre", pre)])

    def _build_weight_vector(self, pipeline: Pipeline) -> np.ndarray:
        pre: ColumnTransformer = pipeline.named_steps["pre"]
        out_names = pre.get_feature_names_out()
        w = np.ones(len(out_names), dtype=np.float32)
        for i, name in enumerate(out_names):
            for src_col, factor in self.cfg.feature_weights.items():
                if name.endswith(f"__{src_col}") or f"__{src_col}_" in name:
                    w[i] *= factor
        return w

    def fit(self, csv_path: str, save: bool = True) -> "CarRecommender":
        raw = pd.read_csv(csv_path)
        self.df = self._prepare_dataframe(raw)

        self.pipeline = self._build_pipeline()
        X = self.pipeline.fit_transform(self.df)
        self._weights_vec = self._build_weight_vector(self.pipeline)
        self._feat_matrix = (X * self._weights_vec).astype(np.float32)

        self.nn = NearestNeighbors(
            n_neighbors=min(self.cfg.n_neighbors + 1, len(self.df)),
            metric=self.cfg.metric, algorithm="brute",
        ).fit(self._feat_matrix)

        if save:
            self.save()
        return self

    def save(self, path: Optional[str] = None) -> str:
        if self.pipeline is None or self.nn is None or self.df is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        out_dir = Path(self.cfg.weights_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = path or str(out_dir / self.cfg.artifact_name)
        joblib.dump({
            "config": asdict(self.cfg),
            "pipeline": self.pipeline,
            "nn": self.nn,
            "df": self.df,
            "weights_vec": self._weights_vec,
            "feat_matrix": self._feat_matrix,
        }, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str) -> "CarRecommender":
        blob = joblib.load(path)
        inst = cls(RecommenderConfig(**blob["config"]))
        inst.pipeline = blob["pipeline"]
        inst.nn = blob["nn"]
        inst.df = blob["df"]
        inst._weights_vec = blob["weights_vec"]
        inst._feat_matrix = blob["feat_matrix"]
        return inst

    def find(self, query: str) -> Optional[int]:
        if self.df is None:
            raise RuntimeError("Model not loaded.")
        key = self._norm(query)
        exact = self.df.index[self.df["_key"] == key]
        if len(exact):
            return int(exact[0])
        partial = self.df.index[self.df["_key"].str.contains(re.escape(key), na=False)]
        return int(partial[0]) if len(partial) else None

    def random_car(self, rng: Optional[np.random.Generator] = None) -> str:
        rng = rng or np.random.default_rng()
        return self.df.iloc[int(rng.integers(0, len(self.df)))]["car_name"]

    def recommend(self, car: str, top_n: int = 5) -> pd.DataFrame:
        idx = self.find(car)
        if idx is None:
            raise ValueError(f"Car '{car}' not found.")
        k = min(self.cfg.n_neighbors + 1, len(self.df))
        dists, neigh = self.nn.kneighbors(self._feat_matrix[idx:idx + 1], n_neighbors=k)
        rows, seen = [], {self.df.iloc[idx]["car_name"]}
        for j, d in zip(neigh[0], dists[0]):
            name = self.df.iloc[j]["car_name"]
            if name in seen:
                continue
            seen.add(name)
            rows.append((j, 1.0 - d))
            if len(rows) >= top_n:
                break
        out = self.df.iloc[[i for i, _ in rows]][list(self.DISPLAY_COLS)].copy()
        out.insert(1, "similarity", [round(s, 4) for _, s in rows])
        return out.reset_index(drop=True)

    def describe(self, car: str) -> pd.Series:
        idx = self.find(car)
        if idx is None:
            raise ValueError(f"Car '{car}' not found.")
        return self.df.iloc[idx][list(self.DISPLAY_COLS)]

    def evaluate(self, sample_size: int = 100, top_n: int = 5,
                 seed: int = 0) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        n = min(sample_size, len(self.df))
        idxs = rng.choice(len(self.df), size=n, replace=False)

        style_hits = style_total = 0
        price_err, hp_err, mpg_err = [], [], []

        for i in idxs:
            src = self.df.iloc[i]
            recs = self.recommend(src["car_name"], top_n=top_n)
            style_hits += int((recs["Vehicle Style"] == src["Vehicle Style"]).sum())
            style_total += len(recs)
            price_err.append(np.abs(recs["MSRP"] - src["MSRP"]) / max(src["MSRP"], 1))
            hp_err.append(np.abs(recs["Engine HP"] - src["Engine HP"])
                          / max(src["Engine HP"], 1))
            mpg_err.append(np.abs(recs["highway MPG"]
                                  if "highway MPG" in recs.columns
                                  else self.df.loc[recs.index, "highway MPG"]
                                  - src["highway MPG"]) / max(src["highway MPG"], 1))

        return {
            f"style_precision@{top_n}": style_hits / max(style_total, 1),
            f"price_mape@{top_n}":      float(np.mean(np.concatenate(price_err))),
            f"hp_mape@{top_n}":         float(np.mean(np.concatenate(hp_err))),
            f"mpg_mape@{top_n}":        float(np.mean(np.concatenate(
                [np.atleast_1d(e) for e in mpg_err]))),
            "n_evaluated": int(n),
        }