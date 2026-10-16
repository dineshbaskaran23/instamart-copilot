import sqlite3
from itertools import combinations
import pandas as pd

pd.set_option("display.width", 200)

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")

done = (o[o["status"] == "completed"]
        .merge(c[["customer_id", "city"]], on="customer_id")
        .merge(p[["product_id", "category"]], on="product_id"))
assert len(done) == 11110

# ---------- (a) pivot cross-check against SQL ----------
pivot = pd.pivot_table(done, values="order_amount", index="city", columns="category",
                       aggfunc="sum", margins=True, margins_name="Total")
print("=== pivot (sum of order_amount) ===")
print(pivot.round(2))
pandas_total = round(pivot.loc["Total", "Total"], 2)

conn = sqlite3.connect("data/instamart.db")
sql_total = conn.execute(
    "SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status = 'completed'").fetchone()[0]
conn.close()
print("\npandas grand total:", pandas_total, "| SQL total:", sql_total)
assert abs(pandas_total - sql_total) < 0.01

print("\n=== per-city sum, mean, count ===")
city = done.groupby("city").agg(total=("order_amount", "sum"),
                                mean=("order_amount", "mean"),
                                n=("order_amount", "count"))
print(city.round(2))

per_cust = done.groupby("customer_id")["order_amount"].sum()
print("\ncustomers with completed spend:", len(per_cust),
      "| average total spend:", round(per_cust.mean(), 2))   # 1175 and 2552.06

# ---------- (b) distribution of completed order_amount ----------
s = done["order_amount"]
q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\n=== order_amount statistics (completed) ===")
print("mean:", round(s.mean(), 2), "| median:", round(q2, 2), "| skew:", round(s.skew(), 2))
print("Q1:", round(q1, 2), "| Q3:", round(q3, 2), "| IQR:", round(iqr, 2))
print("lower fence:", round(lower, 2), "| upper fence:", round(upper, 2))
print("outliers below lower:", int((s < lower).sum()), "| above upper:", int((s > upper).sum()))

# ---------- correlations ----------
cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = done[cols].corr()
print("\n=== correlation matrix ===")
print(corr.round(3))
pairs = {(a, b): corr.loc[a, b] for a, b in combinations(cols, 2)}
for pair, val in sorted(pairs.items(), key=lambda kv: -abs(kv[1])):
    print(pair, round(val, 3))
best = max(pairs, key=lambda k: abs(pairs[k]))
print("largest |correlation|:", best, round(pairs[best], 3))