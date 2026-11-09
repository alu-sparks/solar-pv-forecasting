from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_week7_knn import (
    DATA_PATH,
    FEATURES,
    build_features,
    fit_ridge,
    fit_scaler,
    mae,
    ridge_predict,
    tune_ridge,
)


OUTPUT_DIR = Path("outputs/week8_solar")
RANDOM_SEED = 530


@dataclass
class Node:
    prediction: float
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class RegressionTree:
    def __init__(
        self,
        *,
        max_depth: int,
        min_samples_leaf: int,
        max_features: int,
        rng: np.random.Generator,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng
        self.root: Node | None = None
        self.feature_gain_: np.ndarray | None = None
        self.leaf_count_: int = 0
        self.realized_depth_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RegressionTree":
        self.feature_gain_ = np.zeros(x.shape[1], dtype=float)
        self.leaf_count_ = 0
        self.realized_depth_ = 0
        self.root = self._grow(x, y, depth=0)
        return self

    @staticmethod
    def _sse(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.sum((y - y.mean()) ** 2))

    def _best_split(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[int, float, float] | None:
        n_samples, n_features = x.shape
        candidates = self.rng.choice(
            n_features, size=min(self.max_features, n_features), replace=False
        )
        parent_sse = self._sse(y)
        best: tuple[int, float, float] | None = None

        for feature in candidates:
            order = np.argsort(x[:, feature], kind="mergesort")
            sorted_x = x[order, feature]
            sorted_y = y[order]
            cumulative_y = np.cumsum(sorted_y)
            cumulative_y2 = np.cumsum(sorted_y * sorted_y)

            possible = np.arange(
                self.min_samples_leaf,
                n_samples - self.min_samples_leaf + 1,
                dtype=int,
            )
            if len(possible) == 0:
                continue
            possible = possible[sorted_x[possible - 1] < sorted_x[possible]]
            if len(possible) == 0:
                continue

            # Evaluating at most 32 well-spaced split locations keeps the
            # implementation reproducible and computationally modest.
            if len(possible) > 32:
                positions = np.linspace(0, len(possible) - 1, 32).astype(int)
                possible = possible[positions]

            left_n = possible.astype(float)
            right_n = n_samples - left_n
            left_sum = cumulative_y[possible - 1]
            left_sum2 = cumulative_y2[possible - 1]
            total_sum = cumulative_y[-1]
            total_sum2 = cumulative_y2[-1]
            right_sum = total_sum - left_sum
            right_sum2 = total_sum2 - left_sum2
            left_sse = left_sum2 - (left_sum * left_sum) / left_n
            right_sse = right_sum2 - (right_sum * right_sum) / right_n
            child_sse = left_sse + right_sse
            index = int(np.argmin(child_sse))
            gain = parent_sse - float(child_sse[index])
            split_position = int(possible[index])
            # With a <= split, the lower observed value is a safe threshold.
            # A midpoint can round to the upper value when adjacent floats are
            # extremely close, accidentally leaving an empty right child.
            threshold = float(sorted_x[split_position - 1])
            if gain > 1e-12 and (best is None or gain > best[2]):
                best = (int(feature), threshold, gain)
        return best

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(prediction=float(y.mean()))
        if (
            depth >= self.max_depth
            or len(y) < 2 * self.min_samples_leaf
            or np.allclose(y, y[0])
        ):
            self.leaf_count_ += 1
            self.realized_depth_ = max(self.realized_depth_, depth)
            return node

        split = self._best_split(x, y)
        if split is None:
            self.leaf_count_ += 1
            self.realized_depth_ = max(self.realized_depth_, depth)
            return node

        feature, threshold, gain = split
        left_mask = x[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        assert self.feature_gain_ is not None
        self.feature_gain_[feature] += gain
        node.left = self._grow(x[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(x[~left_mask], y[~left_mask], depth + 1)
        return node

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree must be fit before prediction")
        predictions = np.empty(len(x), dtype=float)
        self._predict_node(self.root, x, np.arange(len(x)), predictions)
        return predictions

    def _predict_node(
        self, node: Node, x: np.ndarray, rows: np.ndarray, out: np.ndarray
    ) -> None:
        if len(rows) == 0:
            return
        if node.feature is None:
            out[rows] = node.prediction
            return
        assert node.threshold is not None and node.left is not None and node.right is not None
        go_left = x[rows, node.feature] <= node.threshold
        self._predict_node(node.left, x, rows[go_left], out)
        self._predict_node(node.right, x, rows[~go_left], out)


class RandomForest:
    def __init__(
        self,
        *,
        n_estimators: int = 120,
        max_depth: int = 7,
        min_samples_leaf: int = 15,
        max_features: int = 4,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.trees: list[RegressionTree] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForest":
        self.trees = []
        n = len(x)
        for _ in range(self.n_estimators):
            sample = self.rng.integers(0, n, size=n)
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                rng=self.rng,
            ).fit(x[sample], y[sample])
            self.trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.mean([tree.predict(x) for tree in self.trees], axis=0)

    def complexity(self) -> dict[str, float]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth_setting": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features_per_split": self.max_features,
            "mean_realized_depth": float(
                np.mean([tree.realized_depth_ for tree in self.trees])
            ),
            "mean_leaves_per_tree": float(
                np.mean([tree.leaf_count_ for tree in self.trees])
            ),
        }

    def impurity_importance(self) -> np.ndarray:
        gains = np.sum([tree.feature_gain_ for tree in self.trees], axis=0)
        total = gains.sum()
        return gains / total if total > 0 else gains


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly = pd.read_csv(DATA_PATH, parse_dates=["measured_on"])
    data = build_features(hourly)
    result_rows: list[dict[str, float | str]] = []
    importance_rows: list[dict[str, float | str]] = []
    complexity_rows: list[dict[str, float | str]] = []

    for month in range(4, 10):
        test_start = pd.Timestamp(2008, month, 1)
        test_end = test_start + pd.offsets.MonthBegin(1)
        train = data.loc[data["measured_on"] < test_start]
        test = data.loc[
            (data["measured_on"] >= test_start)
            & (data["measured_on"] < test_end)
            & data["scheduled_daytime"]
        ]
        train_x = train[FEATURES].to_numpy(float)
        train_y = train["pv_generation_w"].to_numpy(float)
        test_x = test[FEATURES].to_numpy(float)
        test_y = test["pv_generation_w"].to_numpy(float)

        forest = RandomForest(random_state=RANDOM_SEED + month).fit(train_x, train_y)
        forest_prediction = forest.predict(test_x)

        center, scale = fit_scaler(train_x)
        train_x_scaled = (train_x - center) / scale
        test_x_scaled = (test_x - center) / scale
        best_alpha = tune_ridge(train, test_start)
        coefficients = fit_ridge(train_x_scaled, train_y, best_alpha)
        ridge_prediction = ridge_predict(test_x_scaled, coefficients)
        persistence_prediction = test["lag_1"].to_numpy(float)

        predictions = {
            "Random forest": forest_prediction,
            "Ridge": ridge_prediction,
            "Persistence": persistence_prediction,
        }
        mean_actual = float(np.mean(test_y))
        for method, predicted in predictions.items():
            error = mae(test_y, predicted)
            result_rows.append(
                {
                    "test_month": test_start.strftime("%Y-%m"),
                    "method": method,
                    "mae_w": error,
                    "normalized_mae_percent": 100 * error / mean_actual,
                }
            )

        for feature, importance in zip(FEATURES, forest.impurity_importance()):
            importance_rows.append(
                {
                    "test_month": test_start.strftime("%Y-%m"),
                    "feature": feature,
                    "impurity_importance": float(importance),
                }
            )
        complexity_rows.append(
            {
                "test_month": test_start.strftime("%Y-%m"),
                **forest.complexity(),
            }
        )

    results = pd.DataFrame(result_rows)
    summary = (
        results.groupby("method", as_index=False)
        .agg(
            mean_mae_w=("mae_w", "mean"),
            sd_mae_w=("mae_w", "std"),
            mean_normalized_mae_percent=("normalized_mae_percent", "mean"),
        )
        .sort_values("mean_mae_w")
    )
    importances = pd.DataFrame(importance_rows)
    importance_summary = (
        importances.groupby("feature", as_index=False)["impurity_importance"]
        .mean()
        .sort_values("impurity_importance", ascending=False)
    )
    complexity = pd.DataFrame(complexity_rows)
    complexity_summary = {
        "n_estimators": 120,
        "max_depth": 7,
        "min_samples_leaf": 15,
        "max_features_per_split": 4,
        "mean_realized_depth": float(complexity["mean_realized_depth"].mean()),
        "mean_leaves_per_tree": float(complexity["mean_leaves_per_tree"].mean()),
    }
    details = {
        "features": FEATURES,
        "folds": 6,
        "test_months": [f"2008-{month:02d}" for month in range(4, 10)],
        "random_forest_complexity": complexity_summary,
        "summary": summary.to_dict(orient="records"),
        "top_impurity_importances": importance_summary.head(6).to_dict(
            orient="records"
        ),
    }

    results.to_csv(OUTPUT_DIR / "monthly_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
    importances.to_csv(OUTPUT_DIR / "impurity_importance_by_fold.csv", index=False)
    importance_summary.to_csv(OUTPUT_DIR / "impurity_importance_summary.csv", index=False)
    complexity.to_csv(OUTPUT_DIR / "forest_complexity_by_fold.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(importance_summary.head(8).to_string(index=False))
    print(json.dumps(complexity_summary, indent=2))


if __name__ == "__main__":
    main()
