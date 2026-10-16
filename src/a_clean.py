import os, sqlite3
import numpy as np
import pandas as pd

os.makedirs("data/clean", exist_ok=True)

# ---------- customers ----------
cust = pd.read_csv("data/raw/customers.csv")
cust["city"] = cust["city"].str.strip().str.title()

g = cust["gender"].fillna("").str.strip().str.lower()
cust["gender"] = g.map(lambda x: "male" if x.startswith("m")
                       else "female" if x.startswith("f") else "unknown")

cust["age"] = cust["age"].where(cust["age"].between(10, 100))        # outside 10-100 -> NaN
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["signup_date"] = pd.to_datetime(cust["signup_date"])

print("--- customers ---")
print(cust["city"].value_counts())
print(cust["gender"].value_counts())
print("missing age:", cust["age"].isna().sum(), "| min/max age:", cust["age"].min(), cust["age"].max())

# ---------- orders ----------
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")
COLS = list(orders.columns)
print("\n--- orders ---")
print("start rows:", len(orders))

orders["order_ts"] = pd.to_datetime(orders["order_ts"])                              # rule 1
orders = orders.drop_duplicates()                                                   # rule 2
print("after exact duplicates removed:", len(orders))                              # 12965
orders = orders.sort_values("order_ts").drop_duplicates("order_id", keep="last")    # rule 3
print("after latest record per order_id:", len(orders))                            # 12309

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")     # rule 4
orders = orders.merge(products[["product_id", "category", "price_inr"]],
                      on="product_id", how="left")
calc = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
print("amounts filled from price:", orders["order_amount"].isna().sum())           # 120 minus any removed as duplicates
orders["order_amount"] = orders["order_amount"].fillna(calc)

print("negative delivery set to missing:", (orders["delivery_minutes"] < 0).sum())
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan             # rule 5
done = orders["status"] == "completed"
print("completed with missing delivery (to fill):", orders.loc[done, "delivery_minutes"].isna().sum())
med = orders[done].groupby("category")["delivery_minutes"].transform("median")
orders.loc[done, "delivery_minutes"] = orders.loc[done, "delivery_minutes"].fillna(med)

orders = orders[COLS].sort_values("order_id").reset_index(drop=True)
print(orders["status"].value_counts())
print("missing order_amount:", orders["order_amount"].isna().sum())
print("completed missing delivery:", orders.loc[orders["status"] == "completed", "delivery_minutes"].isna().sum())
print("cancelled missing delivery:", orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum())
print("completed amount sum:", round(orders.loc[orders["status"] == "completed", "order_amount"].sum(), 2))

# ---------- store ----------
orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
cust_out = cust.copy()
cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")

cust_out.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders_out.to_csv("data/clean/orders.csv", index=False)

if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
conn = sqlite3.connect("data/instamart.db")
cust_out.to_sql("customers", conn, index=False)
products.to_sql("products", conn, index=False)
orders_out.to_sql("orders", conn, index=False)
conn.close()
print("\nsaved data/clean/*.csv and data/instamart.db")