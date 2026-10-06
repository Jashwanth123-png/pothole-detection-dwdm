# K-Means Clustering Documentation

## 1. What is K-Means?

**K-Means** is an unsupervised machine-learning algorithm that partitions a
dataset into **K groups (clusters)** such that:

- Points within the same cluster are as **similar** to each other as possible.
- Points in different clusters are as **different** as possible.

The algorithm works iteratively:

```
1. Randomly initialise K cluster centroids.
2. Assign each point to the nearest centroid (Euclidean distance).
3. Recompute each centroid as the mean of its assigned points.
4. Repeat steps 2–3 until assignments no longer change (convergence).
```

**K** (the number of clusters) must be specified in advance. We choose K using
the **Elbow Method** and **Silhouette Score** (see below).

---

## 2. Why K-Means for This Project?

Our goal is to identify **pothole hotspots** — geographic concentrations of
frequent or severe potholes — so road repair resources can be prioritised.

| Requirement | Why K-Means works |
|---|---|
| **Hotspot discovery** | Clusters naturally group potholes in similar severity/location zones |
| **Interpretability** | Each cluster has a centroid with mean severity, confidence, size |
| **Speed** | O(n × k × i) — very fast on our dataset size |
| **Simplicity** | Easy to explain to stakeholders and professors |
| **Scikit-learn integration** | Trivial to implement and evaluate |

**Alternatives considered:**

| Algorithm | Why NOT chosen |
|---|---|
| **DBSCAN** | Requires setting `eps` and `min_samples` — hard to tune without lat/long coordinates |
| **Hierarchical clustering** | Computationally expensive (O(n²)); impractical for large datasets |
| **GMM (Gaussian Mixture)** | More complex; probabilistic assignments add ambiguity we don't need |
| **OPTICS** | Good for varying densities, but overkill for our tabular data |

---

## 3. Features Used and Why

The feature matrix sent to K-Means contains:

| Feature | Encoding | Rationale |
|---|---|---|
| `severity_encoded` | High=3, Medium=2, Low=1 | Ordinal — preserves natural severity order |
| `road_type_encoded` | Urban=1, Highway=2, Rural=3 | Ordinal — reflects damage likelihood |
| `confidence` | Raw float [0.6 – 1.0] | Higher confidence = more reliable detections |
| `bbox_area_ratio` | Raw float [0.0 – 1.0] | Proxy for physical pothole size |

**Why these four features?**
- Together they capture **how bad** (severity, size), **how certain** (confidence),
  and **where** (road_type) each detection is.
- We intentionally exclude raw lat/long (not available for all records) and
  `detection_id` (a text key, not a meaningful measure).

**Encoding severity as ordinal (not one-hot):**
- K-Means uses Euclidean distance.
- Ordinal encoding keeps distance proportional: High–Medium distance == Medium–Low distance.
- One-hot encoding would treat severity categories as independent dimensions with
  no order, which is incorrect for this feature.

---

## 4. Normalisation Rationale

Before clustering, all features are standardised using `StandardScaler`:

```
x_scaled = (x - mean) / standard_deviation
```

**Why is this critical?**

Raw features have very different scales:
- `severity_encoded`: values 1, 2, 3.
- `confidence`: values 0.60 to 0.98.
- `bbox_area_ratio`: values 0.01 to 0.15.

Without scaling, `confidence` would dominate the distance metric because its
absolute differences are larger. A point with High severity but low confidence
might be incorrectly grouped with Low-severity points simply because their
confidence values are close.

After `StandardScaler`:
- Every feature has **mean ≈ 0** and **std ≈ 1**.
- All features contribute equally to the Euclidean distance.

---

## 5. Choosing K (Number of Clusters)

### 5.1 Elbow Method

Run K-Means for K = 2, 3, 4, …, 10 and record the **inertia** (Within-Cluster
Sum of Squares — WCSS) for each:

```
WCSS = Σ (distance from each point to its centroid)²
```

Plot WCSS vs K. The optimal K is at the **"elbow"** — the point where adding
more clusters yields diminishing returns.

```
WCSS
│
│\
│  \
│    \
│      ⮑ ← elbow at K=3
│          ─────────────
└──────────────────────── K
     2  3  4  5  6  7
```

For our project, K=3 consistently appears at the elbow, corresponding to
**Low-risk, Medium-risk, and High-risk hotspot zones**.

### 5.2 Default K = 3

We default to K=3 because:
- It maps naturally to our three severity levels.
- Stakeholders (municipalities) can act on three priority tiers: urgent, moderate, monitor.
- The elbow method confirms K=3 for typical city-scale datasets.

The Streamlit UI allows the user to change K (2–8) and observe how clusters shift.

---

## 6. Silhouette Score

The **silhouette score** measures how well each point fits its assigned cluster:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- `a(i)` = mean distance from point i to all other points in the **same** cluster.
- `b(i)` = mean distance from point i to all points in the **nearest other** cluster.

The overall silhouette score is the mean `s(i)` across all points.

| Score | Interpretation |
|---|---|
| 0.71 – 1.00 | Strong cluster structure |
| 0.51 – 0.70 | Reasonable structure |
| 0.26 – 0.50 | Weak structure |
| < 0.25 | No meaningful clusters |

For our synthetic/demo data, typical scores are 0.40 – 0.65, indicating
reasonable but not perfect cluster separation (expected for real-world road data).

---

## 7. Elbow Method (Code)

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertias = []
K_range  = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal K")
plt.savefig("elbow_plot.png")
```

---

## 8. How to Interpret Clusters

After K-Means with K=3, call `get_cluster_summary(result_df)` to get per-cluster
mean statistics:

| cluster_id | avg_severity | avg_confidence | avg_bbox_ratio | count | Interpretation |
|---|---|---|---|---|---|
| 0 | 2.85 | 0.91 | 0.082 | 45 | **High-risk hotspot** — large, high-confidence potholes |
| 1 | 1.62 | 0.73 | 0.031 | 80 | **Medium-risk zone** — moderate potholes, mixed confidence |
| 2 | 1.10 | 0.65 | 0.015 | 35 | **Low-risk area** — small potholes, lower confidence |

**Note:** Cluster numbers are arbitrary (K-Means assigns them randomly each run).
Always interpret via the centroid statistics, not the cluster ID number.

---

## 9. Limitations

| Limitation | Description | Mitigation |
|---|---|---|
| **K must be pre-specified** | No automatic K selection | Use elbow + silhouette; expose K as user parameter |
| **Euclidean distance assumption** | K-Means assumes spherical clusters | Acceptable for our normalised feature space |
| **Sensitive to outliers** | Very large potholes can pull centroids | Could use K-Medoids for robustness |
| **No geographic coordinates** | Clustering on attributes, not lat/long | Add GPS coordinates when available |
| **Random initialisation** | Different `random_state` → different clusters | Use `n_init=10` (default) for stability |
| **Not density-based** | Can't find oddly-shaped clusters | DBSCAN would be better if we had lat/long |
| **No temporal dimension** | Treats all dates equally | Future: add month encoding as a feature |

---

*K-Means documentation — Pothole Detection & Analysis System v1.0*
