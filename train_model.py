from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/Mall_Customers.csv"
MODEL = ROOT / "models/customer_kmeans_model.pkl"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(
        "Put Mall_Customers.csv in data/Mall_Customers.csv"
    )

df = pd.read_csv(DATA)

income_col = "Annual Income (k$)"
score_col = "Spending Score (1-100)"

if income_col not in df.columns or score_col not in df.columns:
    raise ValueError(
        f"Required columns missing. Expected '{income_col}' and '{score_col}'."
    )

X = df[[income_col, score_col]].copy()
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertias = []
ks = range(1, 11)

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(ks), inertias, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "elbow_method.png", dpi=300)
plt.close()

# Five clusters is the standard configuration for this dataset
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

df.to_csv(
    RESULTS / "customer_segments.csv",
    index=False
)

# Save model and scaler
joblib.dump(
    {
        "model": kmeans,
        "scaler": scaler,
        "features": [income_col, score_col]
    },
    MODEL
)

# Convert centroids back to original units for visualization
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(9, 6))

for cluster in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == cluster]
    plt.scatter(
        subset[income_col],
        subset[score_col],
        label=f"Cluster {cluster}"
    )

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "customer_clusters.png", dpi=300)
plt.close()

with open(RESULTS / "model_info.json", "w") as f:
    json.dump({
        "algorithm": "K-Means Clustering",
        "clusters": 5,
        "features": [income_col, score_col],
        "customers": len(df)
    }, f, indent=4)

print("=" * 50)
print("CUSTOMER SEGMENTATION TRAINING COMPLETED")
print("=" * 50)
print(f"Customers: {len(df)}")
print("Clusters: 5")
print(f"Model: {MODEL}")
print("=" * 50)
