import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
done = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]

assert len(o) == 12309
assert o["order_id"].is_unique
assert (o["status"] == "completed").sum() == 11110
assert (o["status"] == "cancelled").sum() == 1199
assert not (o["status"] == "pending").any()
assert o["order_amount"].isna().sum() == 0
assert done["delivery_minutes"].isna().sum() == 0
assert canc["delivery_minutes"].isna().sum() == 1199
assert round(done["order_amount"].sum(), 2) == 2998668.80
assert c["city"].nunique() == 6
assert c["age"].isna().sum() == 0 and c["age"].between(18, 60).all()
assert c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}

conn = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert {"customers", "products", "orders"} <= tables
conn.close()

print("ALL CHECKPOINTS PASSED")