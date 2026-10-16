# Instamart Customer Intelligence & Support Copilot

## Overview
One repository with four connected parts: (A) clean the raw exports into a trusted
SQLite database, (B) segment customers and predict lapse, (C) triage support tickets
with guardrails, (D) a policy RAG assistant and a refund agent. All data is synthetic
and all amounts are in INR. All model calls go through the offline `mock_llm.py`.

## Reproduce
1. `python -m venv .venv`
2. `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python -m src.a_audit`
6. `python -m src.a_clean`
7. `python -m src.a_validate`
8. `python -m src.a_sql_run`
9. `python -m src.a_pandas_eda`
10. `python -m src.b_rfm_cluster`
11. `python -m src.b_lapse`
(Parts B, C and D commands will be added here.)

## Part A

### Audit of the raw data
| Check | Value |
|---|---|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10-100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank gender spellings | 10 |
| Blank gender | 30 |

(Add the shape, data types and missing-value counts per column from `python -m src.a_audit`.)

### Cleaning rules and rows changed
| # | Rule | Effect |
|---|---|---|
| 1 | Convert `order_ts` to datetime | all rows |
| 2 | Remove exact duplicate rows | 119 removed (13,084 to 12,965) |
| 3 | Keep latest `order_ts` per `order_id` | 656 older records removed (12,965 to 12,309) |
| 4 | Unparseable `order_amount` set to missing, filled from price x quantity x (1 - discount/100) | 112 filled |
| 5a | Negative `delivery_minutes` set to missing | 105 values |
| 5b | Completed orders: missing delivery filled with category median (completed orders only) | 774 filled |
| 5c | Cancelled orders: delivery left missing (never delivered) | 1,199 left missing |

Note: the counts in rules 4 and 5a are lower than the raw audit figures (120 and 106)
because some of those bad values sat on duplicate or pending rows that rules 2 and 3
had already removed.

Customers: city stripped and title-cased (6 cities); gender mapped to male, female or
unknown; ages outside 10-100 set to missing, then all missing ages filled with the
median age of the customer's city; `signup_date` converted to datetime.

### Clean data checkpoints
| Check | Value |
|---|---|
| Order rows | 12,309 |
| Completed / cancelled | 11,110 / 1,199 |
| Completed order_amount total | INR 2,998,668.80 |
| Cities | Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184 |
| Genders | female 610, male 560, unknown 30 |

`python -m src.a_validate` prints `ALL CHECKPOINTS PASSED`.

### SQL results (sql/queries.sql)
| Query | Result |
|---|---|
| Q1 | 30 customers |
| Q2 | 30 rows, total INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | 11 products above the average price |
| Q4 | 25 customers with no completed order |
| Q5 | 550 customers above the average spend of INR 2,552.06 |
| Q6 | Top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10. 8 rows have RANK different from ROW_NUMBER. |
| Q7 | 183 daily rows, last cumulative INR 2,998,668.80 |

Why ties change RANK but not ROW_NUMBER: customers with equal spend share one RANK, but
ROW_NUMBER gives every row a different number, so the two differ after a tie.

### pandas cross-check
The pivot table total equals the SQL total (INR 2,998,668.80).

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |
(Add Chennai and Delhi from the script output.)

### Distribution of completed order_amount
| Statistic | Value |
|---|---|
| Mean / median | 269.91 / 135.00 |
| Skewness | 4.61 |
| Q1 / Q3 | 64.80 / 297.00 |
| Lower / upper fence | -283.50 / 645.30 |
| Outliers below / above | 0 / 897 |
| Largest correlation | order_amount and discount_pct, -0.095 |

**Exploratory analysis (max 200 words):**
(Write your own answers to:This data is a record of quick-commerce orders from six Indian cities: groceries, dairy,
snacks, personal care and electronics, with each order's amount, discount, quantity and
delivery time. After cleaning there are 12,309 orders, of which 11,110 were completed.

Order amounts are strongly right-skewed. The mean is INR 269.91 but the median is only
INR 135.00, and the skewness is about 4.61, so most baskets are small and a few are
very large.

Outliers:897 completed orders lie above the upper fence of INR 645.30, and none lie
below the lower fence. 635 of them (about 71%) are Electronics purchases, so they are
genuine high-priced items rather than data errors. The real anomalies were in the raw
data (duplicates, negative delivery times, unreadable amounts), which cleaning fixed.

Relationships are weak. The strongest correlation is between order amount and discount
percentage at about -0.095, just ahead of amount and quantity at +0.087. Correlations
this small are close to no relationship, and they do not establish causation.)

## Part B
(### RFM table
Snapshot 2025-10-01 00:00:00, completed orders only.

| Item | Value |
|---|---|
| Customers with at least one completed order | 1,175 |
| Recency (days): median / min / max | 19 / 0 / 180 |
| Frequency: mean / max | 9.46 / 33 |
| Monetary (INR): mean / max | 2,552.06 / 14,766.55 |

### K-means (StandardScaler, n_init=10, random_state=42)
| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** Inertia falls steeply up to K=4 (1268 to 743) and much more slowly after
it (743 to 617), so the elbow is at K=4. K=4 also has the highest silhouette score (0.526),
and four segments are few enough for the business to run a campaign for each.

**Why K=8 has lower inertia than K=2, yet this cannot pick K:** adding clusters always
brings the centres closer to the points, so inertia keeps falling as K grows, and with one
cluster per customer it would reach 0. A lower number alone therefore says nothing about
whether the groups are useful, which is why the elbow, the silhouette and business
interpretability decide.

### Segments (K=4)
Cluster numbers are only labels and have no ordering.

| Cluster id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 0 | Occasional shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Offer small-basket bundles to raise the order rate |
| 1 | Premium high spenders (Premium) | 138 | 24.12 | 6.77 | 7,129.70 | Early access to electronics offers and priority support |
| 2 | Loyal frequent buyers | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty or subscription programme to keep frequency high |
| 3 | At-risk / lapsing | 182 | 109.04 | 2.20 | 685.99 | Win-back coupon, as the last order was about 109 days ago |

Premium is the segment with the highest mean monetary value.

### PCA (2 components on the scaled RFM matrix)
Explained variance ratio: 0.634 and 0.218, total 0.852.
Output file: `outputs/customer_segments.csv` (1,175 rows).)


## Part C
(to be completed)

## Part D
(to be completed)

## Limitations and Next Steps
(to be completed)