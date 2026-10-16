import json
import os
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

os.makedirs("outputs", exist_ok=True)
pd.set_option("display.width", 200)
random.seed(42)

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATS = ["billing", "delivery", "technical", "fraud"]
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

# ---------- redaction (this exact order) ----------
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\b[6-9]\d{9}\b")
ORDER = re.compile(r"ORD-\d{6}")
STEPS = [(EMAIL, "[EMAIL]"), (PHONE, "[PHONE]"), (ORDER, "[ORDER_ID]")]


def redact(text, counts=None):
    for i, (pat, placeholder) in enumerate(STEPS):
        text, n = pat.subn(placeholder, text)
        if counts is not None:
            counts[i] += n
    return text


# ---------- prompts ----------
ZERO_SHOT = ('You are a support ticket router. CLASSIFY the ticket into exactly one category: '
             'billing, delivery, technical, fraud. Reply with JSON only in the form '
             '{"category": "...", "reason": "..."}.')
EXAMPLES = ('EXAMPLES:\n'
            'Ticket: I was charged twice for [ORDER_ID]. Reply: {"category": "billing", "reason": "double charge"}\n'
            'Ticket: My order has not arrived and the rider is late. Reply: {"category": "delivery", "reason": "late rider"}\n'
            'Ticket: The app freezes and I cannot login. Reply: {"category": "technical", "reason": "app error"}\n'
            'Ticket: Someone hacked my account and used my wallet. Reply: {"category": "fraud", "reason": "unauthorized use"}')
FEW_SHOT = ZERO_SHOT + "\n" + EXAMPLES

assert "CLASSIFY" in ZERO_SHOT and "CLASSIFY" in FEW_SHOT
assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT
print("prompt tokens (estimated): zero-shot", est_tokens(ZERO_SHOT), "| few-shot", est_tokens(FEW_SHOT))

# ---------- retry with exponential backoff and jitter ----------
RETRY_LINES = []


def call_with_retry(messages, ticket_id, stats):
    for retry in range(MAX_RETRIES + 1):
        stats["calls"] += 1
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["transient"] += 1
            if retry == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            RETRY_LINES.append((retry, wait, f"RETRY ticket={ticket_id} retry={retry} wait={wait:.4f}s"))
            time.sleep(wait)


# ---------- output validation ----------
def validate(content):
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = [c for c in CATS if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", None
    if isinstance(obj, dict) and obj.get("category") in CATS:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", None


# ---------- one full pipeline run ----------
def run(system_prompt, records, budget=None):
    mock_llm.reset()
    stats = {"transient": 0, "calls": 0}
    total = 0
    rows = []
    for r in records:
        text = r["text"]
        status, category, used = None, "", 0
        if not text.strip():
            status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif budget is not None and total >= budget:
            status = "budget_exhausted"
        else:
            user_msg = redact(text)
            assert not any(p.search(user_msg) for p, _ in STEPS), "PII reached the prompt"
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}]
            resp = call_with_retry(messages, r["ticket_id"], stats)
            used = resp.usage.total_tokens
            total += used
            status, cat = validate(resp.choices[0].message.content)
            category = cat or ""
        rows.append({"ticket_id": r["ticket_id"], "customer_id": r["customer_id"], "status": status,
                     "category": category, "true_category": r["true_category"], "tokens_used": used})
    return pd.DataFrame(rows), stats


def summarize(label, df, stats):
    cls = df[df["status"].isin(["classified_json", "classified_keyword"])]
    ok = int((cls["category"] == cls["true_category"]).sum())
    total = int(df["tokens_used"].sum())
    print(f"\n=== {label} ===")
    print("statuses:", df["status"].value_counts().to_dict())
    print("transient errors:", stats["transient"], "| model calls incl. retries:", stats["calls"])
    print(f"classified: {len(cls)} | correct: {ok} | accuracy: {ok / len(cls) * 100:.1f}%")
    print("total tokens:", total, "| tokens per classified ticket:", round(total / len(cls), 1))
    return cls


# ---------- routing using Part B segments ----------
seg = pd.read_csv("outputs/customer_segments.csv")
SEG = {r.customer_id: (r.segment_name, str(r.is_premium) == "True") for r in seg.itertuples()}


def add_routing(df):
    names, prios, routes = [], [], []
    for r in df.itertuples():
        name, premium = SEG.get(r.customer_id, ("", False))
        if r.status in HUMAN:
            route, prio = "human_review", "NORMAL"
        elif r.category == "fraud":
            route, prio = "security_team", "HIGH"
        elif premium:
            route, prio = "premium_desk", "HIGH"
        else:
            route, prio = "support_queue", "NORMAL"
        names.append(name); routes.append(route); prios.append(prio)
    out = df.copy()
    out["segment_name"], out["priority"], out["route"] = names, prios, routes
    return out[["ticket_id", "customer_id", "status", "category", "segment_name",
                "priority", "route", "tokens_used"]]


# ================= main =================
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str)
records = tickets.to_dict("records")
print("tickets:", len(records))

# input guardrails
empty = sum(1 for r in records if not r["text"].strip())
long_ = [est_tokens(r["text"]) for r in records if r["text"].strip() and est_tokens(r["text"]) > MAX_INPUT_TOKENS]
print("empty tickets:", empty, "| long tickets:", len(long_), "| long est tokens:", sorted(set(long_)))

# redaction audit over all 164 tickets
counts = [0, 0, 0]
redacted_all = [redact(r["text"], counts) for r in records]
left = sum(len(p.findall(t)) for t in redacted_all for p, _ in STEPS)
print(f"\nredaction: emails {counts[0]} | phones {counts[1]} | order ids {counts[2]} | total {sum(counts)}")
print("PII patterns still found after redaction:", left)
assert left == 0

# zero-shot and few-shot, no budget
df_zero, st_zero = run(ZERO_SHOT, records)
cls_zero = summarize("ZERO-SHOT (no budget)", df_zero, st_zero)
print("\nzero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(cls_zero["true_category"], cls_zero["category"]))

df_few, st_few = run(FEW_SHOT, records)
summarize("FEW-SHOT (no budget)", df_few, st_few)

# budget demo: zero-shot with TOKEN_BUDGET = 6000
DEMO_BUDGET = 6000
df_b, st_b = run(ZERO_SHOT, records, budget=DEMO_BUDGET)
summarize(f"BUDGET DEMO (zero-shot, TOKEN_BUDGET={DEMO_BUDGET})", df_b, st_b)
n_ex = int((df_b["status"] == "budget_exhausted").sum())
print("budget_exhausted tickets:", n_ex)
assert n_ex >= 1, "lower DEMO_BUDGET so at least one ticket is exhausted"
add_routing(df_b).to_csv("outputs/triage_budget_demo.csv", index=False)

# final results: few-shot with TOKEN_BUDGET = 100000
df_f, st_f = run(FEW_SHOT, records, budget=100000)
summarize("FINAL (few-shot, TOKEN_BUDGET=100000)", df_f, st_f)
final = add_routing(df_f)
final.to_csv("outputs/triage_results.csv", index=False)
print("\ntriage_results rows:", len(final))
assert len(final) == 164
print(final.groupby(["route", "priority"]).size())

# retry log check
for retry, wait, _ in RETRY_LINES:
    lo, hi = BASE_DELAY * 2 ** retry * 0.9, BASE_DELAY * 2 ** retry * 1.15
    assert lo <= wait <= hi, "wait outside the jitter range"
with open("outputs/retry_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(line for _, _, line in RETRY_LINES) + "\n")
print("\nretry log lines (all runs):", len(RETRY_LINES), "- every wait is inside its jitter range")
print("first lines:")
for _, _, line in RETRY_LINES[:5]:
    print(" ", line)
    