import json
import os
import shutil

import pandas as pd

from src.d_agent import (SimulatedCrash, load_checkpoint, resume_agent, run_agent)

RUNS = "outputs/agent_runs.jsonl"
DEMOS = "outputs/agent_demos.jsonl"
LEDGER = "outputs/refund_ledger.jsonl"

# start clean, so re-running never duplicates lines
for path in (RUNS, DEMOS, LEDGER):
    if os.path.exists(path):
        os.remove(path)
shutil.rmtree("state", ignore_errors=True)
os.makedirs("outputs", exist_ok=True)


def log(path, res):
    line = {k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used")}
    line["steps"] = res["steps"]
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def refund_of(res):
    for s in res["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0.0


sc = pd.read_csv("data/raw/agent_scenarios.csv", dtype=str, keep_default_na=False)

print("=== seven scenarios ===")
print(f"{'id':<4} {'status':<24} {'refund':>9} {'calls':>5} {'tokens':>7}")
for r in sc.itertuples():
    res = run_agent(r.message, r.scenario_id, run_id=r.scenario_id)
    log(RUNS, res)
    amt = refund_of(res)
    print(f"{r.scenario_id:<4} {res['status']:<24} {amt:>9.2f} {res['llm_calls']:>5} {res['tokens_used']:>7}")
    if r.expected_outcome == "FALLBACK":
        assert res["status"] == "FALLBACK_MAX_ITERATIONS", "S7 must end with FALLBACK_MAX_ITERATIONS"
    else:
        assert res["status"] == r.expected_outcome, (r.scenario_id, res["status"])
        assert abs(amt - float(r.expected_refund_inr)) <= 0.01, (r.scenario_id, amt)
print("all seven scenarios match the expected outcome and refund amount")

s1 = sc.iloc[0]

print("\n=== stop-safely demos (S1) ===")
d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token200")
log(DEMOS, d1)
print("token_limit=200       ->", d1["status"], "| tokens", d1["tokens_used"], "| tools run:", len(d1["steps"]))
assert d1["status"] == "FALLBACK_TOKEN_LIMIT"

d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_timeout0")
log(DEMOS, d2)
print("time_limit_seconds=0  ->", d2["status"], "| model calls", d2["llm_calls"])
assert d2["status"] == "FALLBACK_TIMEOUT"
print("final_answer example:", d2["final_answer"])

print("\n=== crash and resume (S1) ===")
try:
    run_agent(s1.message, "S1", run_id="S1_crash_demo", crash_after_tools=2)
    raise SystemExit("the simulated crash did not happen")
except SimulatedCrash as e:
    print("crashed:", e)
cp = load_checkpoint("S1_crash_demo")
print("checkpoint after crash -> tools executed:", cp["executed"], "| step:", cp["step"])
res = resume_agent("S1_crash_demo")
log(DEMOS, res)
final_cp = load_checkpoint("S1_crash_demo")
print("tools executed across crash and resume:", final_cp["executed"])
print("final status:", res["status"], "| refund:", refund_of(res))
assert final_cp["executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert res["status"] == "REFUND" and abs(refund_of(res) - 128.25) <= 0.01
print("crash-and-resume check passed")