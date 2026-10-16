import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
pd.set_option("display.width", 200)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4          # chosen from {3, 4, 5}; see the K table printed below

# ---------- RFM table ----------
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = o[o["status"] == "completed"]
g = done.groupby("customer_id")
rfm = pd.DataFrame({
    "last_order": g["order_ts"].max(),
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
})
rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()

print("=== RFM ===")
print("customers:", len(rfm))                                                        # 1175
print("recency median/min/max:", rfm["recency"].median(), rfm["recency"].min(), rfm["recency"].max())  # 19, 0, 180
print("frequency mean/max:", round(rfm["frequency"].mean(), 2), rfm["frequency"].max())               # 9.46, 33
print("monetary mean/max:", round(rfm["monetary"].mean(), 2), round(rfm["monetary"].max(), 2))        # 2552.06, 14766.55

# ---------- scale ----------
FEATS = ["recency", "frequency", "monetary"]
X = StandardScaler().fit_transform(rfm[FEATS])

# ---------- K = 2..8 ----------
print("\n=== K-means: inertia and silhouette ===")
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X, km.labels_), 3)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------- final model ----------
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"),
                                     mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"),
                                     mean_monetary=("monetary", "mean")).round(2)

# ---------- name the segments by rule (cluster numbers are only labels) ----------
premium_id = prof["mean_monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["mean_frequency"].idxmax()
rest = rest.drop(index=loyal_id)
names = {premium_id: "Premium high spenders", loyal_id: "Loyal frequent buyers"}
if len(rest) >= 1:
    names[rest["mean_recency"].idxmax()] = "At-risk / lapsing"
if len(rest) == 2:
    names[rest["mean_recency"].idxmin()] = "Occasional shoppers"
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
prof["segment_name"] = prof.index.map(names)

print(f"\n=== segment profile (K={FINAL_K}) ===")
print(prof.to_string())
print("premium segment id:", premium_id)

# ---------- PCA ----------
pca = PCA(n_components=2).fit(X)
ev = pca.explained_variance_ratio_
print("\n=== PCA explained variance ===")
print(ev.round(3), "total:", round(ev.sum(), 3))                                  # 0.634, 0.218, 0.852

# ---------- write output ----------
out = rfm[["customer_id", "recency", "frequency", "monetary",
           "segment_id", "segment_name", "is_premium"]].copy()
out["monetary"] = out["monetary"].round(2)
out.to_csv("outputs/customer_segments.csv", index=False)
print("\nsaved outputs/customer_segments.csv with", len(out), "rows")