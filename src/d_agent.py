import json
import math
import os
import sqlite3
import time
from datetime import datetime, timezone

import mock_llm
from src.d_rag import retrieve

DB = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
STATE_DIR = "state"
AGENT_PROMPT = ("You are a refund AGENT. Use the tools to look up the order, check the policy, "
                "calculate the refund and then either issue it or escalate it to a human. "
                "Finish with a final answer that starts with REFUND, ESCALATE, DENIED or NOT_FOUND.")

_CTX = {"run_id": None}


class SimulatedCrash(Exception):
    pass


# ---------------- tools ----------------
def get_order(order_id):
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders "
                       "WHERE order_id = ?", (order_id,)).fetchone()
    conn.close()
    if row is None:
        return {"error": "order not found"}
    delivery = row[3]
    if delivery is None or (isinstance(delivery, float) and math.isnan(delivery)):
        delivery = None
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": delivery}


def search_policy(query):
    return {"results": retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    line = {"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool, "run_id": _CTX["run_id"],
            "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order by id in the Instamart database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund amount for an order from its amount and delivery time.",
     "parameters": {"type": "object",
                    "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ---------------- checkpoints ----------------
def save_checkpoint(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, default=str)


def load_checkpoint(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------- results ----------------
def _result(st, status, final):
    return {"scenario_id": st["scenario_id"], "run_id": st["run_id"], "status": status,
            "final_answer": final, "steps": st["steps"], "llm_calls": st["llm_calls"],
            "tokens_used": st["tokens_used"]}


def _fallback(st, status, reason):
    done = ", ".join(st["executed"]) or "none"
    final = (f"HANDOFF to a human agent: {reason}. Tools executed so far: {done}. "
             f"Scenario {st['scenario_id']}, model calls {st['llm_calls']}, tokens {st['tokens_used']}.")
    return _result(st, status, final)


# ---------------- ReAct loop ----------------
def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    _CTX["run_id"] = st["run_id"]
    t0 = time.monotonic()
    while True:
        if time.monotonic() - t0 >= time_limit_seconds:
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds} seconds reached")
        if st["llm_calls"] >= max_iterations:
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"no final answer after {max_iterations} model calls")
        resp = mock_llm.chat(st["messages"], TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        msg = resp.choices[0].message
        if st["tokens_used"] > token_limit:                 # stop before executing the requested tool
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit of {token_limit} exceeded")
        if not msg.tool_calls:
            final = msg.content or ""
            return _result(st, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        name, args = tc.name, dict(tc.arguments)
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": name, "arguments": args}]})
        try:
            obs = TOOL_MAP[name](**args)
        except Exception as e:                               # a bad tool call becomes an observation
            obs = {"error": str(e)}
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": json.dumps(obs)})
        st["executed"].append(name)
        st["steps"].append({"tool": name, "arguments": args, "observation": obs})
        st["step"] = len(st["steps"])
        save_checkpoint(st)                                  # after every tool observation
        if crash_after_tools and len(st["executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool execution number {crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": AGENT_PROMPT},
                       {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "executed": [], "steps": []}
    return _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    st = load_checkpoint(run_id)
    return _loop(st, max_iterations, token_limit, time_limit_seconds)