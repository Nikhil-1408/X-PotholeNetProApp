import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL_TO_ID = {"Low": 0, "Medium": 1, "High": 2}
ID_TO_LABEL = {0: "Low", 1: "Medium", 2: "High"}


class SeverityMLModel:
    def __init__(self, csv_path: str = "severity_training.csv") -> None:
        self.feature_names = [
            "area_ratio",
            "darkness",
            "texture",
            "bright_ratio",
            "confidence",
            "model_votes",
            "count_context",
        ]

        self.dt = DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        self.knn = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights="distance")
        )
        self.lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        )

        X, y = self._load_or_build_training_data(csv_path)

        self.dt.fit(X, y)
        self.knn.fit(X, y)
        self.lr.fit(X, y)

    def _load_or_build_training_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            needed = set(self.feature_names + ["label"])
            if needed.issubset(df.columns):
                X = df[self.feature_names].astype(float).values
                y = df["label"].map(LABEL_TO_ID).astype(int).values
                return X, y

        return self._build_synthetic_training_data()

    def _build_synthetic_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        rows: List[List[float]] = []
        labels: List[int] = []

        for _ in range(2400):
            area_ratio = float(rng.uniform(0.001, 0.10))
            darkness = float(rng.uniform(0.05, 0.95))
            texture = float(rng.uniform(0.01, 0.40))
            bright_ratio = float(rng.uniform(0.0, 0.65))
            confidence = float(rng.uniform(0.20, 0.98))
            model_votes = int(rng.integers(1, 3))
            count_context = int(rng.integers(1, 20))

            score = (
                0.58 * min(area_ratio / 0.10, 1.0)
                + 0.10 * min(darkness / 0.80, 1.0)
                + 0.10 * min(texture / 0.30, 1.0)
                + 0.08 * min(confidence / 0.90, 1.0)
                + 0.06 * min(model_votes / 2.0, 1.0)
                + 0.08 * min(count_context / 15.0, 1.0)
                - 0.10 * min(bright_ratio / 0.60, 1.0)
            )

            if area_ratio >= 0.055:
                label = 2
            elif area_ratio >= 0.020:
                label = 2 if score >= 0.72 else 1
            else:
                label = 1 if (score >= 0.62 and confidence >= 0.55 and model_votes >= 2) else 0

            rows.append([
                area_ratio,
                darkness,
                texture,
                bright_ratio,
                confidence,
                model_votes,
                count_context,
            ])
            labels.append(label)

        return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int64)

    def predict(
        self,
        area_ratio: float,
        darkness: float,
        texture: float,
        bright_ratio: float,
        confidence: float,
        model_votes: int,
        count_context: int,
    ) -> Tuple[str, float]:
        x = np.array([[
            area_ratio,
            darkness,
            texture,
            bright_ratio,
            confidence,
            model_votes,
            count_context,
        ]], dtype=np.float32)

        p_dt = self.dt.predict(x)[0]
        p_knn = self.knn.predict(x)[0]
        p_lr = self.lr.predict(x)[0]

        votes = [p_dt, p_knn, p_lr]
        pred = max(set(votes), key=votes.count)

        proba_dt = self.dt.predict_proba(x)[0]
        proba_knn = self.knn.predict_proba(x)[0]
        proba_lr = self.lr.predict_proba(x)[0]
        avg_proba = (proba_dt + proba_knn + proba_lr) / 3.0

        return ID_TO_LABEL[int(pred)], float(avg_proba[int(pred)])