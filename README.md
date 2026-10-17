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
12. `python -m src.c_triage`
13. `python -m src.d_rag`
14. `python -m src.d_demos`

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
Output file: `outputs/customer_segments.csv` (1,175 rows).
### Lapse problem
Cutoff 2025-09-01. Cohort: customers with a completed order before the cutoff.
`lapsed` = 1 if there is no completed order from 2025-09-01 to 2025-09-30.

| Item | Value |
|---|---|
| Cohort size | 1,156 |
| Lapsed | 443 (38.32%) |
| Train / test rows | 924 / 232 |
| Baseline (always "not lapsed"): accuracy / lapsed F1 | 0.616 / 0.0 |

Features: recency, frequency, monetary, average discount, cancellation rate, average
delivery minutes, favourite category, city, age, gender, tenure in days.

**No-leakage statements:** no order dated on or after 2025-09-01 entered the features; the
segment labels from the clustering were not used as features; no encoder, scaler or
resampler saw test rows (encoder and scaler sit inside a scikit-learn Pipeline fitted on
training rows, and the over-sampler was applied to training rows only). No feature column
has a missing value.

Scaler mean of frequency (training rows): 8.115 vs full-cohort mean 8.029. They differ
because the scaler learned its mean from the 924 training rows only, not all 1,156
customers.

### Tuning and test results
| Model | Best parameters | Best CV F1 |
|---|---|---|
| Logistic regression | C = 100 | 0.651 |
| Random forest | max_depth = 8, min_samples_leaf = 10 | 0.681 |

Test set, lapsed class (confusion matrix as [[TN, FP], [FN, TP]]):

| Model | Precision | Recall | F1 | Confusion matrix |
|---|---|---|---|---|
| Baseline | 0 | 0 | 0 | [[143, 0], [89, 0]] |
| Logistic regression | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Random forest | 0.659 | 0.652 | 0.655 | [[113, 30], [31, 58]] |

**Selected model: random forest.** It had the higher cross-validated F1 (0.681 against
0.651) and the slightly higher test F1 (0.655 against 0.648). Both models find the same 58
of the 89 lapsing customers, and the forest raises 30 false alarms against 32. Missing a
customer who is about to lapse loses their future orders, while wrongly targeting a loyal
customer only costs a small coupon, so recall matters more than precision here.

### Class-balance experiment
Training rows only were over-sampled with replacement (random_state=42) to 570 / 570 per
class, then the best logistic regression was refitted. The test set was unchanged.

| Logistic model | Precision | Recall | F1 |
|---|---|---|---|
| Plain | 0.644 | 0.652 | 0.648 |
| Balanced | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.202), while precision fell slightly (-0.055). The balanced model
catches many more lapsing customers, but about 4 in 10 of the customers it flags are loyal
ones, so the retention team gets a bigger list with more wasted offers. That is acceptable
when a coupon is cheap compared with a lost customer.

### Interpretation
Top 5 random-forest features by importance: frequency 0.280, monetary 0.271, recency
0.115, average delivery minutes 0.061, cancellation rate 0.052. How often and how much a
customer buys are the strongest signals of whether they stay; long delivery times and
cancellations also matter.

Logistic coefficient signs (standardised features):
- frequency: negative (-1.698). More orders before the cutoff means lower lapse risk.
- recency: positive (0.040). A longer gap since the last order slightly raises lapse
  risk, but the effect is small.
- cancel_rate: negative (-0.124). A small effect that I would not read much into.

### Saved model and retention list
`outputs/lapse_model.joblib` was reloaded with joblib.load and reproduced the original
test predictions exactly. `outputs/retention_list.csv` lists the top 10 Premium test-set
customers by predicted lapse probability; 1 of the 10 truly lapsed. The predicted
probabilities in the list are all below 0.5 (0.452 to 0.491), so the model is not
confident about any of these customers. This list is illustrative and not a
leakage-free evaluation, because the Premium flag comes from whole-window RFM that
includes September, the window that defines the label.)
## Part C
(### Input guardrails and redaction
Before any model call, empty or whitespace-only tickets get `skipped_empty` and tickets
with an estimated length above MAX_INPUT_TOKENS = 400 get `flagged_long`. Neither is ever
sent to the model.

| Item | Value |
|---|---|
| Empty tickets | 4 |
| Long tickets | 4 (each about 734 estimated tokens) |

Personal data is redacted with these patterns, in this order, before a prompt is built:
email to `[EMAIL]`, phone `\b[6-9]\d{9}\b` to `[PHONE]`, order id `ORD-\d{6}` to `[ORDER_ID]`.
The user message sent to the model is exactly the redacted ticket text.

| Type | Substitutions |
|---|---|
| Emails | 55 |
| Phone numbers | 91 |
| Order ids | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted texts finds 0 matches, and the code asserts
that no pattern matches any prompt before each call.

### Prompts
Both prompts contain the word CLASSIFY, list the four categories (billing, delivery,
technical, fraud) and require a JSON-only reply of the form
`{"category": "...", "reason": "..."}`. The zero-shot prompt has no `EXAMPLES:` line (37
estimated tokens). The few-shot prompt adds an `EXAMPLES:` line with four labelled examples,
including a fraud one (115 estimated tokens).

### Retries
Every model call goes through a retry function: at most 3 retries per ticket, wait before
retry n = 0.05 x 2^n x random.uniform(0.9, 1.15) seconds, one log line per retry. Each full
run over the 156 eligible tickets raised exactly 19 transient errors and all were recovered
by a retry, so no ticket ended with a permanent error. Every retry was retry number 0, with
waits between 0.0453 and 0.0542 seconds (allowed range 0.045 to 0.0575). The log is in
`outputs/retry_log.txt` (68 lines over the four runs).

### Output validation
A reply is accepted as `classified_json` if it is valid JSON with an allowed category. If it
is not valid JSON, the lowercase words are extracted with `re.findall(r"[a-z]+", ...)` and
the reply is accepted as `classified_keyword` if exactly one allowed category word is
present. Everything else, including valid JSON with a category outside the list, is
`flagged_invalid_output` and goes to a human. Nothing is guessed.

| Status (one full run) | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Zero-shot versus few-shot
| Run | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| Zero-shot | 147 | 126 | 85.7% | 10,180 | 69.3 |
| Few-shot | 147 | 147 | 100.0% | 22,348 | 152.0 |

Zero-shot confusion table (rows = true category, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

(Points to use: the examples cost about 83 extra
tokens per classified ticket, 152.0 against 69.3, so the whole run used 12,168 more tokens;
in return accuracy rose by 14.3 points, from 85.7% to 100%. Every zero-shot error was a
fraud ticket (11 of 34) or a technical ticket (10 of 39) sent to billing. A fraud ticket
sent to billing misses the high-priority security route, so the extra tokens are worth
paying here, while for low-stakes labels the cheaper zero-shot prompt would be enough.)

### Token budget demo
Zero-shot run with TOKEN_BUDGET = 6000: the budget is checked before every call, and once
the running total reached 6000 no further call was made. 64 tickets ended `budget_exhausted`
(routed to human review); 87 were classified before the budget ran out (6,010 tokens
used). Result: `outputs/triage_budget_demo.csv`.

### Routing and final output
`outputs/triage_results.csv` (164 rows) comes from a few-shot run with TOKEN_BUDGET =
100000, using the segment file from Part B. A customer with no row in the segment file is
treated as not premium.

| Route | Priority | Tickets |
|---|---|---|
| human_review | NORMAL | 17 |
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |

**Rule-validated vs human:** 147 tickets (135 JSON + 12 keyword) passed the validation rules
and were routed automatically by category and segment. 17 tickets went to human review: 4
empty, 4 too long and 9 with an invalid model answer. )

## Part D
(### Tools
Five tools are defined in `TOOLS_SPEC` (name, one-sentence description, JSON-style
parameters) and mapped by name in `TOOL_MAP`. The model asks for a tool by name with an
arguments dictionary, and the code looks the name up and calls the function.

| Tool | What it does |
|---|---|
| get_order(order_id) | Reads `data/instamart.db`; returns order_id, status, order_amount, delivery_minutes, or `{"error": "order not found"}` |
| search_policy(query) | Returns the top 3 chunks from the TF-IDF retrieval under `results` |
| calculate_refund(order_amount, delivery_minutes) | 100% of the amount if delivery took more than 45 minutes, otherwise 0 |
| issue_refund(order_id, refund_inr) | Appends a line (timestamp, tool name) to `outputs/refund_ledger.jsonl` |
| escalate_to_human(order_id, refund_inr) | Appends a line to the same ledger; used for refunds above INR 1000 |

### ReAct loop
`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
time_limit_seconds=10)` starts with a system message containing AGENT and the user message.
Each iteration calls the model. If the reply has no tool call, its content is the final
answer and the status is the word before the colon. Otherwise the assistant tool-call
message is appended, the tool is run, and a tool message with the JSON result is appended.
It returns status, final_answer, steps, llm_calls and tokens_used.

### Seven scenarios
| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | ___ |
| S2 | REFUND | 135.00 | 5 | ___ |
| S3 | ESCALATE | 1999.00 (sent to a human, not issued) | 5 | ___ |
| S4 | DENIED | 0.00 | 4 | ___ |
| S5 | DENIED | 0.00 | 3 | ___ |
| S6 | NOT_FOUND | 0.00 | 2 | ___ |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | ___ |

Every refund amount equals `expected_refund_inr` within 0.01. S7 uses a loop trigger in the
message, finishes all six model calls under the default token limit of 6000, and then stops.
One line per run is in `outputs/agent_runs.jsonl`.

### Stop safely
- **Max iterations:** after 6 model calls without a final answer the run stops with
  `FALLBACK_MAX_ITERATIONS` (scenario S7).
- **Token limit:** after each model call the tokens are added up, and if the total is above
  the limit the run stops at once, without running the tool that was just requested
  (`FALLBACK_TOKEN_LIMIT`).
- **Time limit:** at the top of each iteration, if `time.monotonic()` shows that the limit
  or more has passed, the run stops (`FALLBACK_TIMEOUT`).

Every fallback returns the steps completed so far and a short hand-off message in
`final_answer`.

| Demo (scenario S1) | Status |
|---|---|
| token_limit = 200 | FALLBACK_TOKEN_LIMIT (___ tokens used, ___ tools executed) |
| time_limit_seconds = 0 | FALLBACK_TIMEOUT (0 model calls) |

### Crash and resume
After every tool observation a checkpoint is written to `state/<run id>.json` with the run
id, scenario id, the full message list, tokens used, the step number and the tools already
executed. For S1 a simulated crash was raised right after the second tool execution (after
its checkpoint was saved), and `resume_agent(run_id)` loaded the checkpoint and continued.

Tools executed across the crash and the resume, in order:
1. get_order
2. search_policy
3. calculate_refund
4. issue_refund

That is four executions with none repeated, and the final status was REFUND with INR 128.25.
Demo runs are in `outputs/agent_demos.jsonl`. The ledger holds the S1 refund twice, once
from the S1 scenario run and once from the crash-and-resume demo; the lines carry different
`run_id` values.

### ReAct versus plan-and-execute
My ReAct agent asks the model what to do after every observation: S1 took 5 model calls, S5
took 3, S6 took 2, and S7 used all 6 before the limit stopped it. A plan-and-execute design
would ask the model once for the whole step list and run the tools from code, so a normal
refund like S1 would need about 1 or 2 model calls instead of 5. It would also use fewer
tokens, because the growing message history, including the policy text from the search,
would not be sent back to the model after every tool.

The difference shows with unexpected observations. With ReAct, a NOT_FOUND order (S6) or a
cancelled order (S5) is seen by the model straight away, and it finishes after 2 or 3 calls.
A plan written in advance would still contain the refund steps, so the code needs a check
after every step and one extra model call to replan when the result is unexpected.

For this refund workflow I would choose plan-and-execute with replanning only on error. The
rules are fixed (look up, check status, compute, issue or escalate), so the plan is
predictable and cheaper, and I would keep the step, token and time limits and the
checkpoints. ReAct would be the better choice if the workflow were open-ended.)

## Limitations and Next Steps
( The Premium flag is a post-hoc whole-window label (its RFM includes September, the window
  that defines the lapse label), so the retention list is illustrative and not a
  leakage-free evaluation.
- All data is synthetic, and the language model is an offline mock that classifies by
  keywords, so the 100% few-shot accuracy says more about the mock than about a real model.
- The lapse test set has only 232 customers, so the F1 scores are noisy, and the model
  could be improved with more behavioural features and probability calibration.
- TF-IDF only matches exact words; embeddings would handle paraphrases better.
- The agent handles one order per message and treats every request as inside the 7-day
  window.
- Next steps: evaluate on real tickets, add a human-in-the-loop review screen for escalated
  refunds, and add monitoring for token cost and error rates.)