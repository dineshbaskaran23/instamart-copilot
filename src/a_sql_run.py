import re, sqlite3
import pandas as pd

pd.set_option("display.width", 200)
conn = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d)\s*$", text, flags=re.M)

results = {}
for name, sql in zip(parts[1::2], parts[2::2]):
    df = pd.read_sql_query(sql, conn)
    results[name] = df
    print("=====", name, "| rows:", len(df))
    print(df.head(8).to_string(index=False))

print("\n--- checkpoints ---")
print("Q1 rows (30):", len(results["Q1"]))
print("Q2 rows (30):", len(results["Q2"]), "| total (2998668.80):", round(results["Q2"]["revenue"].sum(), 2))
print("Q2 top cell:", results["Q2"].iloc[0].to_dict())
print("Q3 rows (11):", len(results["Q3"]))
print("Q4 rows (25):", len(results["Q4"]))
print("Q5 rows (550):", len(results["Q5"]))
q6 = results["Q6"]
print("Q6 rows where RANK != ROW_NUMBER (8):", int((q6["rnk"] != q6["rn"]).sum()))
print("Q6 top per city:")
print(q6[q6["rn"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
q7 = results["Q7"]
print("Q7 rows (183):", len(q7), "| last cumulative (2998668.80):", round(q7["cumulative_revenue"].iloc[-1], 2))
conn.close()