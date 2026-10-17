import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = "data/policies"

SYSTEM_PROMPT = ("You are a POLICY ASSISTANT. Answer only from the supplied context. "
                 "If the answer is not in the context, say that you could not find the "
                 "information in the policy documents.")


def chunk_words(words, size=40, overlap=10):
    """Chunks of `size` words; each next chunk starts size-overlap words later.
    Stops after the first chunk whose end reaches or passes the last word."""
    step = size - overlap
    chunks, start = [], 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


class Index:
    def __init__(self, size=40, overlap=10):
        self.size, self.overlap = size, overlap
        self.chunks = []
        for fname in sorted(os.listdir(POLICY_DIR)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(POLICY_DIR, fname), encoding="utf-8") as f:
                words = f.read().split()
            for text in chunk_words(words, size, overlap):
                self.chunks.append({"source": fname, "text": text})
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        q = self.vec.transform([query])                       # same fitted vectorizer
        sims = cosine_similarity(q, self.matrix)[0]
        order = np.argsort(-sims, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"],
                 "score": round(float(sims[i]), 4),
                 "text": self.chunks[i]["text"]} for i in order]


DEFAULT = Index(40, 10)


def retrieve(query, k=3):
    return DEFAULT.retrieve(query, k)


def answer(question, index=None):
    index = index or DEFAULT
    chunks = index.retrieve(question, 3)
    context = "\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    user = "CONTEXT:\n" + context + "\nQUESTION:\n" + question
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}])
    return resp.choices[0].message.content, [c["source"] for c in chunks]


def evaluate(ev, index):
    n_ans = n_unans = doc_hits = fact_hits = fallbacks = 0
    problems = []
    for r in ev.itertuples():
        ans, sources = answer(r.question, index)
        if r.expected_doc == "NONE":
            n_unans += 1
            if ans.startswith("I could not find"):
                fallbacks += 1
            else:
                problems.append((r.question, "expected the fallback answer", ans))
        else:
            n_ans += 1
            d = r.expected_doc in sources
            f = r.expected_fact in ans                       # case-sensitive substring
            doc_hits += d
            fact_hits += f
            if not (d and f):
                problems.append((r.question, f"doc in top 3: {d} | fact in answer: {f}", ans))
    return n_ans, doc_hits, fact_hits, n_unans, fallbacks, problems


if __name__ == "__main__":
    from collections import Counter

    for label, idx in [("overlap 10", DEFAULT), ("overlap 0", Index(40, 0))]:
        c = Counter(ch["source"] for ch in idx.chunks)
        print(f"chunks ({label}): total {len(idx.chunks)} |", dict(c))

    top = retrieve("refund policy late delivery")
    print("\nquery 'refund policy late delivery' -> top 3 sources:", [t["source"] for t in top])
    print("scores:", [t["score"] for t in top])

    txt, src = answer("Who must approve a refund above INR 1000?")
    print("\nsample answer:", txt, "| sources:", src)

    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False, dtype=str)
    print("\nquestions:", len(ev))
    for label, idx in [("OVERLAP 10", DEFAULT), ("OVERLAP 0", Index(40, 0))]:
        n_ans, dh, fh, n_un, fb, problems = evaluate(ev, idx)
        print(f"\n=== {label} ===")
        print(f"answerable: expected doc in top 3 = {dh}/{n_ans} | expected fact in answer = {fh}/{n_ans}")
        print(f"unanswerable: fallback answer = {fb}/{n_un}")
        for q, why, a in problems:
            print("  PROBLEM:", q, "|", why)
            print("     answer:", a)