"""
mining/kmeans.py
================
K-Means Clustering for pothole hotspot detection.

This implements the K-Means algorithm from scikit-learn on warehouse data.

IMPORTANT NOTES:
  1. Cluster labels are NEUTRAL: Cluster 0, Cluster 1, Cluster 2, etc.
     We do NOT automatically call them 'Low Risk', 'Medium Risk', 'High Risk'
     unless a formal project-defined mapping is configured.
  2. Features are normalized using StandardScaler before clustering.
  3. K (number of clusters) is configurable via config.yaml (default: 3).
  4. Characteristics of each cluster are described using feature averages.

Data Mining Context:
  K-Means groups roads/locations with similar pothole patterns.
  This helps identify road maintenance priorities and hotspots.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from mining.prepare_features import get_features_for_clustering
from utils.helpers import load_config, setup_logger

logger = setup_logger(__name__)


class PotholeKMeans:
    """
    K-Means clustering for pothole hotspot analysis.

    Clusters roads/locations based on six pothole-related features and exposes
    helper methods to inspect cluster quality, centers, and per-cluster summaries.

    Attributes:
        n_clusters (int): Number of clusters K.
        random_state (int): Random seed for reproducibility.
        max_iter (int): Maximum K-Means iterations.
        kmeans (KMeans | None): Fitted sklearn KMeans object (set after fit()).
        df (pd.DataFrame | None): Input DataFrame with cluster assignments appended.
        X_scaled (np.ndarray | None): Normalized feature matrix used for fitting.
        scaler (StandardScaler | None): Fitted StandardScaler used for normalization.
        feature_names (list | None): Names of feature columns in matrix order.
        is_fitted (bool): True once fit() has been called successfully.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42, max_iter: int = 300):
        """
        Initialize PotholeKMeans with hyperparameters.

        Args:
            n_clusters: Number of clusters K (default: 3).
            random_state: Random seed for reproducible results (default: 42).
            max_iter: Maximum number of K-Means iterations (default: 300).
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

        # Set after calling fit()
        self.kmeans = None
        self.df = None
        self.X_scaled = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False

        logger.info(f"PotholeKMeans initialized: K={n_clusters}, "
                    f"random_state={random_state}, max_iter={max_iter}")

    # ------------------------------------------------------------------
    # Core fit method
    # ------------------------------------------------------------------

    def fit(self, session=None) -> pd.DataFrame:
        """
        Load warehouse data, prepare features, and fit K-Means.

        Automatically reduces K if fewer records are available than requested
        clusters (e.g., small demo dataset).

        Args:
            session: Optional SQLAlchemy database session. If None, a new
                     session is opened internally.

        Returns:
            DataFrame with all feature columns plus 'cluster_id' (int) and
            'cluster_label' (str, e.g. "Cluster 0") columns appended.

        Raises:
            ValueError: If fewer than 2 records exist in the warehouse.
        """
        # Step 1 – Load and normalize features from the data warehouse
        self.df, self.X_scaled, self.scaler, self.feature_names = (
            get_features_for_clustering(session)
        )

        # Step 2 – Guard: K cannot exceed the number of data points
        actual_k = min(self.n_clusters, len(self.df))
        if actual_k != self.n_clusters:
            logger.warning(
                f"Reduced K from {self.n_clusters} to {actual_k} "
                f"(only {len(self.df)} records available)"
            )
            self.n_clusters = actual_k

        # Step 3 – Fit scikit-learn KMeans
        logger.info(
            f"Fitting K-Means: K={self.n_clusters}, records={len(self.df)}"
        )
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=10,              # compatible with sklearn 1.3+
        )
        cluster_labels = self.kmeans.fit_predict(self.X_scaled)

        # Step 4 – Attach cluster columns to the DataFrame
        self.df = self.df.copy()
        self.df["cluster_id"] = cluster_labels
        self.df["cluster_label"] = [f"Cluster {c}" for c in cluster_labels]

        self.is_fitted = True
        logger.info(
            f"K-Means complete. Inertia={self.kmeans.inertia_:.2f}, "
            f"Silhouette={self.get_silhouette_score():.3f}"
        )
        return self.df

    # ------------------------------------------------------------------
    # Inspection methods
    # ------------------------------------------------------------------

    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Return a summary table with size and mean feature values per cluster.

        Cluster characteristics are described by the mean value of each feature
        within the cluster. This is intentionally neutral – no risk labels are
        assigned here because cluster ordering is arbitrary in K-Means.

        Returns:
            DataFrame where each row is one cluster, columns:
              cluster_id, cluster_label, cluster_size, mean_<feature>...

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before get_cluster_summary()")

        summary_rows = []
        for cid in range(self.n_clusters):
            cluster_data = self.df[self.df["cluster_id"] == cid]

            row = {
                "cluster_id": cid,
                "cluster_label": f"Cluster {cid}",
                "cluster_size": len(cluster_data),
            }

            # Mean of each raw (unscaled) feature column
            for feat in self.feature_names:
                if feat in cluster_data.columns:
                    row[f"mean_{feat}"] = round(float(cluster_data[feat].mean()), 3)

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def get_silhouette_score(self) -> float:
        """
        Compute the silhouette score to evaluate clustering quality.

        The silhouette score measures how similar each point is to its own
        cluster compared to other clusters. Values closer to +1 are better;
        values near 0 indicate overlapping clusters; negative values suggest
        misclassification.

        Returns:
            Float in [-1, 1], or -1.0 if the score cannot be computed
            (e.g., only 1 cluster exists).
        """
        if not self.is_fitted or self.n_clusters < 2:
            logger.debug("Silhouette score not computable (n_clusters < 2 or not fitted)")
            return -1.0

        try:
            score = silhouette_score(self.X_scaled, self.df["cluster_id"].values)
            return float(score)
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            return -1.0

    def get_cluster_centers_df(self) -> pd.DataFrame:
        """
        Return cluster centers transformed back to the original feature scale.

        Because features are normalized before clustering, the raw
        KMeans.cluster_centers_ are in scaled space. This method applies the
        inverse StandardScaler transform so values are interpretable.

        Returns:
            DataFrame with one row per cluster, columns:
              cluster_label, <feature_name>...

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before get_cluster_centers_df()")

        # Inverse-transform from scaled space to original feature space
        centers_original = self.scaler.inverse_transform(self.kmeans.cluster_centers_)

        centers_df = pd.DataFrame(centers_original, columns=self.feature_names)
        centers_df.insert(
            0, "cluster_label", [f"Cluster {i}" for i in range(self.n_clusters)]
        )
        return centers_df.round(3)

    def to_dict(self) -> dict:
        """
        Serialize all clustering results to a plain Python dictionary.

        Suitable for JSON export (used by cluster_analysis.export_results)
        and for populating the dashboard.

        Returns:
            Dict with keys: n_clusters, n_records, inertia, silhouette_score,
            feature_names, cluster_summary, cluster_centers.
        """
        if not self.is_fitted:
            return {"error": "Model has not been fitted yet. Call fit() first."}

        return {
            "n_clusters": self.n_clusters,
            "n_records": len(self.df),
            "inertia": round(float(self.kmeans.inertia_), 4),
            "silhouette_score": round(self.get_silhouette_score(), 4),
            "feature_names": self.feature_names,
            "cluster_summary": self.get_cluster_summary().to_dict(orient="records"),
            "cluster_centers": self.get_cluster_centers_df().to_dict(orient="records"),
        }


# ---------------------------------------------------------------------------
# Convenience runner function
# ---------------------------------------------------------------------------

def run_kmeans(
    n_clusters: int = None,
    config: dict = None,
    session=None,
) -> PotholeKMeans:
    """
    Build, configure, and fit a PotholeKMeans model using project config.

    Config keys read from config.yaml under the 'kmeans' section:
      - n_clusters   (default: 3)
      - random_state (default: 42)
      - max_iter     (default: 300)

    Args:
        n_clusters: Override for number of clusters (takes priority over config).
        config: Project config dict. If None, loaded from config.yaml automatically.
        session: Optional SQLAlchemy session.

    Returns:
        Fitted PotholeKMeans instance ready for analysis.
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            logger.warning("Could not load config.yaml – using defaults")
            config = {}

    km_cfg = config.get("kmeans", {})

    # n_clusters argument takes priority; fall back to config, then default of 3
    k = n_clusters or km_cfg.get("n_clusters", 3)
    random_state = km_cfg.get("random_state", 42)
    max_iter = km_cfg.get("max_iter", 300)

    model = PotholeKMeans(n_clusters=k, random_state=random_state, max_iter=max_iter)
    model.fit(session)
    return model


# ---------------------------------------------------------------------------
# Quick demo when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== K-MEANS CLUSTERING ===")

    try:
        model = run_kmeans()

        print(f"\nClusters:         {model.n_clusters}")
        print(f"Records:          {len(model.df)}")
        print(f"Inertia:          {model.kmeans.inertia_:.2f}")
        print(f"Silhouette Score: {model.get_silhouette_score():.3f}")

        print("\n--- Cluster Summary ---")
        print(model.get_cluster_summary().to_string(index=False))

        print("\n--- Cluster Centers (original scale) ---")
        print(model.get_cluster_centers_df().to_string(index=False))

        print("\n--- Sample Records with Cluster Assignments ---")
        cols = ["road_name", "city", "cluster_label", "pothole_count", "detection_frequency"]
        available = [c for c in cols if c in model.df.columns]
        print(model.df[available].to_string(index=False))

    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Run: python warehouse/seed_demo_data.py")
