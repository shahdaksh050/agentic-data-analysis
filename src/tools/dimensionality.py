"""
Dimensionality Analysis Tool — Execution Layer.

Stage 3: PCA variance structure and multicollinearity screening for
datasets with many numeric features (DatasetProfile.is_high_dimensional).

Answers two questions a data scientist asks before modelling wide data:
  - How many components capture most of the variance? (PCA)
  - Which feature pairs are redundant enough to cause instability? (|r| > 0.9)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.tools.base import BaseTool, ToolExecutionError
from src.tools.clustering import _select_cluster_features
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: |correlation| at/above this between two features is reported as redundant.
HIGH_CORRELATION_THRESHOLD = 0.9


class DimensionalityAnalysisTool(BaseTool):
    """PCA explained-variance structure plus a multicollinearity screen."""

    requires_ml = True

    name = "dimensionality_analysis"
    description = (
        "Analyse a wide numeric feature space: PCA explained variance per component "
        "(and how many components reach the variance_threshold), plus feature pairs "
        "with |correlation| > 0.9 (multicollinearity risk). Use when the data profile "
        "shows is_high_dimensional (many numeric features)."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if profile is not None and profile.is_high_dimensional else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str | None = None,
        variance_threshold: float = 0.95,
        **_: Any,
    ) -> dict[str, Any]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        df = _read_df(file_path)
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])
        features = _select_cluster_features(df)

        if features.shape[1] < 2:
            raise ToolExecutionError(
                "Dimensionality analysis needs at least 2 usable numeric features; "
                f"found {features.shape[1]}."
            )

        filled = features.fillna(features.median(numeric_only=True))

        # ---- Multicollinearity screen ----
        corr = filled.corr(method="pearson")
        cols = corr.columns.tolist()
        high_corr_pairs: list[dict[str, Any]] = []
        for i, ca in enumerate(cols):
            for cb in cols[i + 1:]:
                val = float(corr.loc[ca, cb])
                if not np.isnan(val) and abs(val) >= HIGH_CORRELATION_THRESHOLD:
                    high_corr_pairs.append({"col_a": ca, "col_b": cb, "correlation": round(val, 4)})
        high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # ---- PCA ----
        scaler = StandardScaler()
        X = scaler.fit_transform(filled)
        n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)

        explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        cumulative = np.cumsum(explained)
        n_for_threshold = int(np.searchsorted(cumulative, variance_threshold) + 1)
        n_for_threshold = min(n_for_threshold, len(explained))

        return {
            "summary": (
                f"{features.shape[1]} numeric features → {n_for_threshold} PCA component(s) "
                f"explain {variance_threshold:.0%} of variance. "
                f"{len(high_corr_pairs)} feature pair(s) with |r| ≥ {HIGH_CORRELATION_THRESHOLD}."
            ),
            "n_features": int(features.shape[1]),
            "features_used": list(features.columns),
            "explained_variance_ratio": explained,
            "cumulative_variance": [round(float(v), 4) for v in cumulative],
            "n_components_for_threshold": n_for_threshold,
            "variance_threshold": variance_threshold,
            "high_correlation_pairs": high_corr_pairs,
            "multicollinearity_risk": bool(high_corr_pairs),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the (cleaned) dataset.", "required": True},
            "target_column": {
                "type": "string",
                "description": "Excluded from the feature space if given.",
                "required": False,
            },
            "variance_threshold": {
                "type": "float",
                "description": "Cumulative variance fraction to report component count for. Default: 0.95.",
                "required": False,
            },
        }
