#!/usr/bin/env python
"""Score the returned neutral-instruction relevance re-run against relevance_key.csv.
Reports the motif-relevant 'yes' rate on the 48 name-match pairs (vs the primed
annotators' 47/48 and 44/48) and the distractor rejection rate, so the paper can state
whether the check survives without theme-priming. Usage:
  python score_neutral_eval.py [path/to/returned relevance_eval_neutral.csv]"""
import os, sys, csv
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "neutral_annotator_packet", "relevance_eval_neutral.csv")
key = {r["pair_id"]: r["kind"] for r in csv.DictReader(open(f"{DATA}/human_eval/relevance_key.csv"))}
ans = {r["pair_id"]: (r["relevant_yes_no"] or "").strip().lower() for r in csv.DictReader(open(path))}

filled = sum(1 for v in ans.values() if v in ("yes", "no"))
nm = [p for p, k in key.items() if k == "relevant_namematch"]
dc = [p for p, k in key.items() if k == "attention_check_expect_no"]
nm_yes = sum(1 for p in nm if ans.get(p) == "yes")
nm_judged = sum(1 for p in nm if ans.get(p) in ("yes", "no"))
dc_no = sum(1 for p in dc if ans.get(p) == "no")
dc_judged = sum(1 for p in dc if ans.get(p) in ("yes", "no"))

def wilson(k, n, z=1.96):
    if not n: return (float("nan"),)*2
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/d
    return max(0, c-h), min(1, c+h)

print(f"answers filled: {filled}/60")
lo, hi = wilson(nm_yes, nm_judged)
print(f"\nNAME-MATCH (relevant) pairs: {nm_yes}/{nm_judged} judged YES "
      f"({nm_yes/nm_judged:.0%}, Wilson 95% CI [{lo:.2f},{hi:.2f}])")
print(f"  primed-annotator reference: 47/48 (98%) and 44/48 (92%)")
print(f"DISTRACTOR pairs: {dc_no}/{dc_judged} correctly judged NO ({dc_no/dc_judged:.0%})")
print(f"  primed-annotator reference: 11/12 (92%) and 12/12 (100%)")
verdict = "SUPPORTS motif-relevance without priming" if nm_yes/max(1,nm_judged) >= 0.75 else "WEAKER under neutral framing — down-scope the claim"
print(f"\n=> {verdict}")
