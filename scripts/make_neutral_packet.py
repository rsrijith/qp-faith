#!/usr/bin/env python
"""Generate a NEUTRAL-instruction relevance packet for the annotator re-run requested
by the final review (the prior packet primed annotators toward query-theme over
category). Same 60 pairs as relevance_key.csv, blanked answers, shuffled, with
instructions that do NOT tell the judge how to resolve the motif-vs-category question.
Output: neutral_annotator_packet/{INSTRUCTIONS.md, relevance_eval_neutral.csv}."""
import os, csv, random
import pandas as pd
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "neutral_annotator_packet")
os.makedirs(OUT, exist_ok=True)

key = pd.read_csv(f"{DATA}/human_eval/relevance_key.csv")
# titles come from the existing eval file, which carries all 60 pairs verbatim
src = pd.read_csv(f"{DATA}/human_eval/relevance_eval.csv")
title_of = dict(zip(src.pair_id, src.product_title))
query_of = dict(zip(src.pair_id, src["query"]))

rows = []
for pid in key["pair_id"]:
    q = query_of.get(pid) or pid.rsplit("-", 1)[0]
    title = title_of.get(pid, "")
    assert title, f"no title for {pid}"
    rows.append({"pair_id": pid, "query": q, "product_title": title, "relevant_yes_no": ""})

random.Random(2026).shuffle(rows)  # fixed-seed shuffle so order differs from the primed packet
with open(f"{OUT}/relevance_eval_neutral.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pair_id", "query", "product_title", "relevant_yes_no"])
    w.writeheader()
    for r in rows: w.writerow(r)

INSTR = """# Relevance check (about 15 minutes)

Thank you for helping. This is a simple relevance-judgment task for a research paper on
online shopping search. No special background is needed.

## What to do
Open `relevance_eval_neutral.csv` in any spreadsheet app. Each row has a **search query**
that someone typed into a shopping website and a **product title**. For every row, put
**yes** or **no** in the `relevant_yes_no` column, answering one question:

> Would a shopper who typed this query reasonably want this product shown in the results?

## How to judge
- Judge naturally, the way you would as a real shopper. There are no special rules to follow.
- Use your own honest interpretation of what the shopper is looking for. Do not overthink it.
- If you genuinely cannot tell, leave the cell blank. Otherwise please fill every row.

## Important
- Please judge **on your own**, without discussing the items with anyone else first.
  Independent answers are the point.
- There are 60 rows.

When done, save the file (keep the name `relevance_eval_neutral.csv`) and send it back. Thank you.
"""
open(f"{OUT}/INSTRUCTIONS.md", "w").write(INSTR)
print(f"wrote {OUT}/ : relevance_eval_neutral.csv ({len(rows)} rows) + INSTRUCTIONS.md")
print("note: no theme-vs-category priming; this is the neutral re-run the paper says would strengthen the check.")
