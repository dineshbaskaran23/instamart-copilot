import pandas as pd

customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print("=====", name, df.shape)
    print(df.dtypes)
    print(pd.DataFrame({"missing": df.isna().sum(),
                        "pct": (df.isna().mean() * 100).round(2)}))

print("\n--- orders audit ---")
print("rows:", len(orders))                                                    # 13084
print("exact duplicates:", orders.duplicated().sum())                          # 119
print("pending rows:", (orders["status"] == "pending").sum())                  # 656
print("distinct order_id:", orders["order_id"].nunique())                      # 12309
print("non-numeric amounts:", pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum())  # 120
print("negative delivery:", (orders["delivery_minutes"] < 0).sum())            # 106
print("blank delivery:", orders["delivery_minutes"].isna().sum())              # 2541

print("\n--- customers audit ---")
print("blank age:", customers["age"].isna().sum())                             # 61
print("age outside 10-100:", ((customers["age"] < 10) | (customers["age"] > 100)).sum())  # 6
print("city spellings:", customers["city"].nunique())                          # 24
print("gender spellings:", customers["gender"].nunique())                      # 10
print("blank gender:", customers["gender"].isna().sum())                       # 30