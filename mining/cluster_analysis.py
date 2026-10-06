"""
mining/cluster_analysis.py
==========================
Analysis and reporting tools for K-Means clustering results.

Provides higher-level utilities built on top of PotholeKMeans:

  - analyze_clusters()    : Detailed per-cluster statistics and interpretations.
  - find_elbow()          : Inertia values for K=1..max_k (elbow-method data).
  - describe_cluster()    : Human-readable text description of a cluster.
  - get_hotspots()        : Top-N hotspot roads from the highest-severity cluster.
  - export_results()      : Save full clustering results to a JSON file.

Usage:
  from mining.cluster_analysis import analyze_clusters, find_elbow, get_hotspots
  from mining.kmeans import run_kmeans

  model = run_kmeans()
  report = analyze_clusters(model)
  hotspots = get_hotspots(model, top_n=5)
  export_results(model, "results/clustering_output.json")
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent))

from mining.prepare_features import get_features_for_clustering
from utils.helpers import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# analyze_clusters
# ---------------------------------------------------------------------------

def analyze_clusters(kmeans_model) -> dict:
    """
    Produce a detailed analysis dictionary for every cluster.

    For each cluster the analysis includes:
      - size              : Number of roads/locations in the cluster.
      - percentage        : Share of total records (%).
      - feature_means     : Mean value of each raw (unscaled) feature.
      - feature_stds      : Standard deviation of each raw feature.
      - top_roads         : Up to 5 roads with the highest pothole_count.
      - description       : Human-readable text summary (from describe_cluster).

    Args:
        kmeans_model (PotholeKMeans): A *fitted* PotholeKMeans instance.

    Returns:
        dict with structure::

            {
              "n_clusters":   int,
              "n_records":    int,
              "silhouette":   float,
              "inertia":      float,
              "clusters": {
                  0: { "size": ..., "percentage": ..., "feature_means": {...}, ... },
                  1: { ... },
                  ...
              }
            }

    Raises:
        RuntimeError: If kmeans_model has not been fitted.
    """
    if not kmeans_model.is_fitted:
        raise RuntimeError("kmeans_model must be fitted before calling analyze_clusters()")

    df = kmeans_model.df
    feature_names = kmeans_model.feature_names
    n_total = len(df)

    clusters_info = {}

    for cid in range(kmeans_model.n_clusters):
        cluster_df = df[df["cluster_id"] == cid]
        size = len(cluster_df)

        # --- Feature statistics (raw, unscaled values) ---
        feature_means = {}
        feature_stds = {}
        for feat in feature_names:
            if feat in cluster_df.columns:
                feature_means[feat] = round(float(cluster_df[feat].mean()), 3)
                feature_stds[feat] = round(float(cluster_df[feat].std(skipna=True)), 3)

        # --- Top roads by pothole count ---
        sort_col = "pothole_count" if "pothole_count" in cluster_df.columns else cluster_df.columns[0]
        top_roads = (
            cluster_df[["road_name", "city", sort_col]]
            .sort_values(sort_col, ascending=False)
            .head(5)
            .to_dict(orient="records")
        )

        # --- Human-readable description ---
        description = describe_cluster(cluster_df, feature_names)

        clusters_info[cid] = {
            "cluster_id": cid,
            "cluster_label": f"Cluster {cid}",
            "size": size,
            "percentage": round(size / n_total * 100, 1) if n_total > 0 else 0.0,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "top_roads": top_roads,
            "description": description,
        }

    result = {
        "n_clusters": kmeans_model.n_clusters,
        "n_records": n_total,
        "silhouette": round(kmeans_model.get_silhouette_score(), 4),
        "inertia": round(float(kmeans_model.kmeans.inertia_), 4),
        "clusters": clusters_info,
    }

    logger.info(f"Cluster analysis complete: {kmeans_model.n_clusters} clusters, "
                f"silhouette={result['silhouette']}")
    return result


# ---------------------------------------------------------------------------
# find_elbow
# ---------------------------------------------------------------------------

def find_elbow(max_k: int = 10, session=None) -> pd.DataFrame:
    """
    Compute K-Means inertia for K = 1 to max_k to support the elbow method.

    The elbow method is used to choose an optimal K: plot inertia vs K and
    look for the "elbow" where adding more clusters gives diminishing returns.

    Args:
        max_k: Maximum number of clusters to try (default: 10).
        session: Optional SQLAlchemy session.

    Returns:
        DataFrame with columns ['k', 'inertia'] suitable for plotting.

    Example::

        elbow_df = find_elbow(max_k=8)
        print(elbow_df)
        # k  inertia
        # 1  450.3
        # 2  210.1
        # ...
    """
    logger.info(f"Running elbow method for K=1..{max_k}")

    # Load features once; reuse for all K values
    try:
        df, X_scaled, scaler, feature_names = get_features_for_clustering(session)
    except ValueError as e:
        logger.error(f"Cannot run elbow method: {e}")
        return pd.DataFrame(columns=["k", "inertia"])

    # Cap max_k at number of available records to avoid sklearn errors
    max_k = min(max_k, len(df))

    rows = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, max_iter=300, n_init="auto")
        km.fit(X_scaled)
        inertia = round(float(km.inertia_), 4)
        rows.append({"k": k, "inertia": inertia})
        logger.debug(f"  K={k}: inertia={inertia}")

    elbow_df = pd.DataFrame(rows)
    logger.info("Elbow method complete")
    return elbow_df


# ---------------------------------------------------------------------------
# describe_cluster
# ---------------------------------------------------------------------------

def describe_cluster(cluster_df: pd.DataFrame, feature_names: list) -> str:
    """
    Generate a human-readable text description of a single cluster.

    Interprets the mean feature values and produces a plain-English summary.
    This avoids hard-coding risk labels, instead describing the cluster
    objectively based on its average characteristics.

    Args:
        cluster_df: Subset of the main DataFrame belonging to one cluster.
        feature_names: List of feature column names used in clustering.

    Returns:
        A multi-sentence string describing the cluster's characteristics.

    Example return::

        "This cluster contains 12 roads. Average pothole count is HIGH (mean=45.3).
         Detection frequency is MODERATE (mean=8.1). Severity score is HIGH (2.7/3).
         Roads in this cluster appear on 6 distinct dates on average.
         Confidence in detections is HIGH (mean=0.87). Road type is mostly Urban."
    """
    if cluster_df.empty:
        return "Empty cluster – no records assigned."

    size = len(cluster_df)
    lines = [f"This cluster contains {size} road/location record(s)."]

    # Helper: classify a normalized ratio into Low / Moderate / High
    def _level(value: float, low_thresh: float, high_thresh: float) -> str:
        if value <= low_thresh:
            return "LOW"
        elif value <= high_thresh:
            return "MODERATE"
        else:
            return "HIGH"

    # --- Pothole count ---
    if "pothole_count" in cluster_df.columns:
        mean_pc = float(cluster_df["pothole_count"].mean())
        level = _level(mean_pc, 5, 20)
        lines.append(
            f"Average pothole count is {level} (mean={mean_pc:.1f} potholes)."
        )

    # --- Detection frequency ---
    if "detection_frequency" in cluster_df.columns:
        mean_df_val = float(cluster_df["detection_frequency"].mean())
        level = _level(mean_df_val, 3, 8)
        lines.append(
            f"Detection frequency is {level} (mean={mean_df_val:.1f} events)."
        )

    # --- Severity score ---
    if "severity_score" in cluster_df.columns:
        mean_sev = float(cluster_df["severity_score"].mean())
        # Severity is 1=Low, 2=Medium, 3=High
        sev_label = (
            "Low (mostly minor potholes)"
            if mean_sev < 1.67
            else "Medium (mixed severity)"
            if mean_sev < 2.34
            else "High (predominantly severe potholes)"
        )
        lines.append(
            f"Severity is {sev_label} (score={mean_sev:.2f}/3.0)."
        )

    # --- Historical occurrence (unique dates) ---
    if "historical_occurrence" in cluster_df.columns:
        mean_ho = float(cluster_df["historical_occurrence"].mean())
        level = _level(mean_ho, 2, 6)
        lines.append(
            f"Roads appear on {level} number of distinct detection dates "
            f"(mean={mean_ho:.1f} dates)."
        )

    # --- Confidence ---
    if "avg_confidence" in cluster_df.columns:
        mean_conf = float(cluster_df["avg_confidence"].mean())
        conf_label = _level(mean_conf, 0.5, 0.75)
        lines.append(
            f"Detection confidence is {conf_label} (mean={mean_conf:.2f})."
        )

    # --- Road type ---
    if "road_type" in cluster_df.columns:
        most_common_rt = cluster_df["road_type"].mode()
        if not most_common_rt.empty:
            lines.append(f"Most common road type in this cluster: {most_common_rt.iloc[0]}.")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# get_hotspots
# ---------------------------------------------------------------------------

def get_hotspots(kmeans_model, top_n: int = 10) -> pd.DataFrame:
    """
    Identify the top-N hotspot roads from the cluster with the highest
    average pothole count.

    The "hotspot cluster" is defined as the cluster whose mean pothole_count
    is the highest. Within that cluster, roads are ranked by pothole_count.

    Args:
        kmeans_model (PotholeKMeans): A fitted PotholeKMeans instance.
        top_n: Number of top hotspot roads to return (default: 10).

    Returns:
        DataFrame of up to top_n rows with columns including road_name,
        city, area, cluster_label, pothole_count, detection_frequency,
        severity_score, and avg_confidence.

    Raises:
        RuntimeError: If kmeans_model has not been fitted.

    Example::

        hotspots = get_hotspots(model, top_n=5)
        print(hotspots[["road_name", "city", "pothole_count"]].to_string())
    """
    if not kmeans_model.is_fitted:
        raise RuntimeError("kmeans_model must be fitted before calling get_hotspots()")

    df = kmeans_model.df

    if "pothole_count" not in df.columns:
        logger.warning("pothole_count column not found; returning top records by cluster size")
        return df.head(top_n)

    # Find the cluster with the highest mean pothole_count
    cluster_means = (
        df.groupby("cluster_id")["pothole_count"]
        .mean()
        .sort_values(ascending=False)
    )
    hotspot_cluster_id = int(cluster_means.index[0])

    logger.info(
        f"Hotspot cluster identified: Cluster {hotspot_cluster_id} "
        f"(mean pothole_count={cluster_means.iloc[0]:.1f})"
    )

    # Filter to hotspot cluster, rank by pothole_count
    hotspot_df = (
        df[df["cluster_id"] == hotspot_cluster_id]
        .sort_values("pothole_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Select meaningful columns for display (keep only those that exist)
    preferred_cols = [
        "road_name", "area", "city",
        "cluster_label", "pothole_count",
        "detection_frequency", "severity_score",
        "avg_confidence", "road_type",
    ]
    display_cols = [c for c in preferred_cols if c in hotspot_df.columns]
    return hotspot_df[display_cols]


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------

def export_results(kmeans_model, output_path: str) -> str:
    """
    Export complete clustering results to a JSON file.

    The exported JSON contains:
      - Model metadata (n_clusters, inertia, silhouette score).
      - Cluster summary table (size + mean features per cluster).
      - Cluster centers in original feature scale.
      - Detailed cluster analysis (from analyze_clusters()).
      - Full per-road cluster assignments.

    Args:
        kmeans_model (PotholeKMeans): A fitted PotholeKMeans instance.
        output_path: Destination file path (e.g., "results/kmeans_results.json").
                     Parent directories are created automatically.

    Returns:
        Absolute path of the written JSON file as a string.

    Raises:
        RuntimeError: If kmeans_model has not been fitted.
        OSError: If the file cannot be written.

    Example::

        path = export_results(model, "mining/output/results.json")
        print(f"Saved to: {path}")
    """
    if not kmeans_model.is_fitted:
        raise RuntimeError("kmeans_model must be fitted before calling export_results()")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build full export payload
    analysis = analyze_clusters(kmeans_model)

    # Convert per-road assignments to list of dicts
    df = kmeans_model.df.copy()
    # Convert numpy types to native Python types for JSON serialization
    for col in df.select_dtypes(include=[np.integer]).columns:
        df[col] = df[col].astype(int)
    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = df[col].astype(float)

    road_assignments = df.to_dict(orient="records")

    export_data = {
        "model_metadata": {
            "n_clusters": kmeans_model.n_clusters,
            "n_records": len(df),
            "inertia": round(float(kmeans_model.kmeans.inertia_), 4),
            "silhouette_score": round(kmeans_model.get_silhouette_score(), 4),
            "random_state": kmeans_model.random_state,
            "max_iter": kmeans_model.max_iter,
            "feature_names": kmeans_model.feature_names,
        },
        "cluster_summary": kmeans_model.get_cluster_summary().to_dict(orient="records"),
        "cluster_centers": kmeans_model.get_cluster_centers_df().to_dict(orient="records"),
        "cluster_analysis": analysis,
        "road_assignments": road_assignments,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, default=str)

    abs_path = str(output_path.resolve())
    logger.info(f"Results exported to: {abs_path}")
    return abs_path


# ---------------------------------------------------------------------------
# Demo when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Import here to avoid circular import at module level
    from mining.kmeans import run_kmeans

    print("\n=== CLUSTER ANALYSIS DEMO ===")

    try:
        # Step 1: Fit the model
        print("\n[1] Fitting K-Means model...")
        model = run_kmeans()
        print(f"    Fitted: {model.n_clusters} clusters on {len(model.df)} records")
        print(f"    Silhouette score: {model.get_silhouette_score():.3f}")

        # Step 2: Analyze clusters
        print("\n[2] Analyzing clusters...")
        analysis = analyze_clusters(model)
        for cid, info in analysis["clusters"].items():
            print(f"\n  --- Cluster {cid} ({info['size']} roads, {info['percentage']}%) ---")
            print(f"  {info['description']}")
            if info["top_roads"]:
                print(f"  Top roads: {[r['road_name'] for r in info['top_roads']]}")

        # Step 3: Elbow method
        print("\n[3] Running elbow method (K=1..6)...")
        elbow_df = find_elbow(max_k=6)
        print(elbow_df.to_string(index=False))

        # Step 4: Hotspots
        print("\n[4] Top hotspot roads...")
        hotspots = get_hotspots(model, top_n=5)
        if not hotspots.empty:
            cols = [c for c in ["road_name", "city", "pothole_count", "cluster_label"]
                    if c in hotspots.columns]
            print(hotspots[cols].to_string(index=False))
        else:
            print("    No hotspots found.")

        # Step 5: Export results
        print("\n[5] Exporting results to JSON...")
        out_dir = Path(__file__).parent / "output"
        out_path = export_results(model, str(out_dir / "kmeans_results.json"))
        print(f"    Saved: {out_path}")

    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Run: python warehouse/seed_demo_data.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
