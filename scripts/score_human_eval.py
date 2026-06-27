#!/usr/bin/env python
"""Score the filled human relevance-validation sheet.
Validation stat = fraction of motif-matched (probe-relevant) products the human judged relevant;
high => the probe's relevant set matches human relevance => the measured harm is real.
Also: attention-check pass rate, and Cohen's kappa if a 2nd annotator file exists."""
import csv, os
D=os.path.join(os.path.dirname(__file__),"..","data","human_eval")
def load(p):
    return {r["pair_id"]: (r.get("relevant_yes_no") or "").strip().lower() for r in csv.DictReader(open(p))}
key={r["pair_id"]: r["kind"] for r in csv.DictReader(open(f"{D}/relevance_key.csv"))}
ann=load(f"{D}/relevance_eval.csv")

nm=[pid for pid,k in key.items() if k=="relevant_namematch"]
ac=[pid for pid,k in key.items() if k=="attention_check_expect_no"]
yes=lambda pid: ann.get(pid,"").startswith("y")
no=lambda pid: ann.get(pid,"").startswith("n")
nm_filled=[p for p in nm if ann.get(p)]
ac_filled=[p for p in ac if ann.get(p)]
print(f"motif-matched products judged RELEVANT: {sum(yes(p) for p in nm_filled)}/{len(nm_filled)} "
      f"= {sum(yes(p) for p in nm_filled)/max(1,len(nm_filled)):.0%}  "
      f"(high => probe relevant-set matches human relevance => harm is real)")
print(f"attention checks judged NOT relevant: {sum(no(p) for p in ac_filled)}/{len(ac_filled)} "
      f"= {sum(no(p) for p in ac_filled)/max(1,len(ac_filled)):.0%}  (should be high)")

a2p=f"{D}/relevance_eval_annotator2.csv"
if os.path.exists(a2p):
    a2=load(a2p)
    both=[p for p in ann if ann.get(p) and a2.get(p)]
    if both:
        agree=sum(ann[p][0]==a2[p][0] for p in both)/len(both)
        # Cohen's kappa
        po=agree
        pa1=sum(ann[p].startswith("y") for p in both)/len(both); pa2=sum(a2[p].startswith("y") for p in both)/len(both)
        pe=pa1*pa2+(1-pa1)*(1-pa2); kappa=(po-pe)/(1-pe) if pe<1 else 1.0
        print(f"inter-annotator: raw agreement {agree:.0%} on {len(both)} overlap, Cohen's kappa {kappa:.2f}")
else:
    print("(no 2nd-annotator file; single-annotator validation only)")
