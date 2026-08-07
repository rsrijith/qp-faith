#!/usr/bin/env python
"""R1.5: build a HARD type-less probe from the 55 A_entity motifs, adding the harder query
classes Reviewer 1 named: misspellings and compositional queries. Every variant is still
type-less by construction (no product category named), so the correct plan emits no
product_type; a category emission is a false application, exactly as on the base probe.
Deterministic (no RNG). Usage: ./.venv/bin/python scripts/make_hard_probe.py"""
import json, os
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

motifs = [json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl")]
motifs = [m for m in motifs if m.get("family") == "A_entity"]

def misspell(w):
    # deterministic realistic typo: transpose the two middle characters
    if len(w) < 4:
        return w + w[-1]           # short words: double the last char
    i = len(w) // 2
    return w[:i-1] + w[i] + w[i-1] + w[i+1:]

MODIFIERS = ["vintage", "large", "colorful", "antique", "giant"]  # attributes, not product types

rows = []
for j, m in enumerate(motifs):
    q = m["query"]
    # 1) misspelled
    rows.append({"query": misspell(q), "variant": "misspelled", "base": q,
                 "family": "A_entity", "known_slots": {}, "relevant_pids": m.get("relevant_pids", [])})
    # 2) compositional: entity + entity (pair with the next motif, wrap around)
    other = motifs[(j + 1) % len(motifs)]["query"]
    rows.append({"query": f"{q} and {other}", "variant": "compositional_entity", "base": q,
                 "family": "A_entity", "known_slots": {}, "relevant_pids": m.get("relevant_pids", [])})
    # 3) compositional: attribute modifier + entity (still no product type)
    mod = MODIFIERS[j % len(MODIFIERS)]
    rows.append({"query": f"{mod} {q}", "variant": "modifier", "base": q,
                 "family": "A_entity", "known_slots": {}, "relevant_pids": m.get("relevant_pids", [])})

out = f"{DATA}/probe/probe_hard.jsonl"
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
from collections import Counter
c = Counter(r["variant"] for r in rows)
print(f"wrote {len(rows)} hard queries to {out}: {dict(c)}")
print("samples:", [r["query"] for r in rows[:6]])
