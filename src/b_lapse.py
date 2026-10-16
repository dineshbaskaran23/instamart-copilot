import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("outputs", exist_ok=True)
pd.set_option("display.width", 200)

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

# ---------- data ----------
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
o = o.merge(p[["product_id", "category"]], on="product_id", how="left")

before = o[o["order_ts"] < CUTOFF]                       # ONLY orders before the cutoff feed the features
done_b = before[before["status"] == "completed"]
assert done_b["order_ts"].max() < CUTOFF

# ---------- label: no completed order in September 2025 ----------
sept = o[(o["status"] == "completed") & (o["order_ts"] >= CUTOFF) & (o["order_ts"] < END)]
active = set(sept["customer_id"])

# ---------- features ----------
g = done_b.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
})
canc = before[before["status"] == "cancelled"].groupby("customer_id").size()
tot = before[before["status"].isin(["completed", "cancelled"])].groupby("customer_id").size()
feat["cancel_rate"] = (canc.reindex(feat.index).fillna(0) / tot.reindex(feat.index)).fillna(0)
fav = (done_b.groupby(["customer_id", "category"]).size().reset_index(name="n")
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id")["category"])
feat["fav_category"] = fav
feat = feat.reset_index().merge(c[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")
feat["lapsed"] = (~feat["customer_id"].isin(active)).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[NUM + CAT], feat["lapsed"]
assert X.isna().sum().sum() == 0
print("no missing values in features: True")
print("cohort:", len(feat), "| lapsed:", int(y.sum()), f"({y.mean()*100:.2f}%)")          # 1156, 443, 38.32%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train/test:", len(X_train), len(X_test))                                           # 924 / 232

def metrics(yt, yp):
    return (round(precision_score(yt, yp, zero_division=0), 3),
            round(recall_score(yt, yp, zero_division=0), 3),
            round(f1_score(yt, yp, zero_division=0), 3))

# ---------- baseline ----------
base = np.zeros(len(y_test), dtype=int)
print("\nBASELINE (always not lapsed): accuracy", round(accuracy_score(y_test, base), 3),
      "| lapsed F1", round(f1_score(y_test, base, zero_division=0), 3))

# ---------- pipeline: encoder and scaler live INSIDE, so they only ever fit on training rows ----------
def make_pipe(clf):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])

lr_gs = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)),
                     {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr_gs.fit(X_train, y_train)
rf_gs = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                     {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1")
rf_gs.fit(X_train, y_train)

print("\nLOGISTIC best:", lr_gs.best_params_, "| CV F1", round(lr_gs.best_score_, 3))
print("FOREST   best:", rf_gs.best_params_, "| CV F1", round(rf_gs.best_score_, 3))

# ---------- scaler leakage check ----------
scaler = lr_gs.best_estimator_.named_steps["pre"].named_transformers_["num"]
print("\nscaler mean of frequency (train only):", round(scaler.mean_[NUM.index("frequency")], 3),
      "| full-cohort mean:", round(feat["frequency"].mean(), 3))

# ---------- test-set evaluation (after tuning is finished) ----------
print("\n=== TEST SET (lapsed class) ===")
for name, gs in [("logistic", lr_gs), ("forest", rf_gs)]:
    pred = gs.best_estimator_.predict(X_test)
    pr, rc, f1 = metrics(y_test, pred)
    print(name, "precision", pr, "recall", rc, "F1", f1)
    print(confusion_matrix(y_test, pred))

best_gs = rf_gs if rf_gs.best_score_ >= lr_gs.best_score_ else lr_gs      # chosen by CV score, not by the test set
best_name = "forest" if best_gs is rf_gs else "logistic"
best_model = best_gs.best_estimator_
print("\nselected model (higher CV F1):", best_name)

# ---------- class-balance experiment: oversample TRAINING rows only ----------
n_need = int((y_train == 0).sum() - (y_train == 1).sum())
extra = X_train[y_train == 1].sample(n=n_need, replace=True, random_state=42)
X_bal = pd.concat([X_train, extra])
y_bal = pd.concat([y_train, pd.Series(1, index=extra.index)])
print("\nbalanced train counts:", y_bal.value_counts().to_dict())                          # 570 each
lr_plain = lr_gs.best_estimator_
lr_bal = clone(lr_plain).fit(X_bal, y_bal)
print("logistic plain   (P, R, F1):", metrics(y_test, lr_plain.predict(X_test)))
print("logistic balanced(P, R, F1):", metrics(y_test, lr_bal.predict(X_test)))

# ---------- interpretation ----------
rf_best = rf_gs.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\ntop 5 forest importances:")
print(imp.head(5).round(3))
lr_names = lr_plain.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr_plain.named_steps["clf"].coef_[0], index=lr_names)
print("\nlogistic coefficients (numeric features):")
print(coef[[n for n in lr_names if n.startswith("num__")]].round(3))

# ---------- persist and reload ----------
joblib.dump(best_model, "outputs/lapse_model.joblib")
loaded = joblib.load("outputs/lapse_model.joblib")
assert np.array_equal(loaded.predict(X_test), best_model.predict(X_test))
print("\nreloaded model reproduces the test predictions exactly: True")

# ---------- retention list (premium customers in the test set) ----------
seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"] == True, "customer_id"])
t = pd.DataFrame({"customer_id": feat.loc[X_test.index, "customer_id"].values,
                  "lapse_probability": best_model.predict_proba(X_test)[:, 1].round(4),
                  "actually_lapsed": y_test.values})
ret = (t[t["customer_id"].isin(premium_ids)]
       .sort_values("lapse_probability", ascending=False).head(10))
ret.to_csv("outputs/retention_list.csv", index=False)
print("\nretention list rows:", len(ret), "| truly lapsed:", int(ret["actually_lapsed"].sum()))
print(ret.to_string(index=False))