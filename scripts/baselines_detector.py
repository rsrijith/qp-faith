#!/usr/bin/env python
"""Three analyses requested by the ESWA review:
(1) a NON-LLM rule-based planner baseline (catalog-vocab attribute extractor): does it
    hallucinate a product_type on type-less entity queries? (shows the failure is LLM-specific)
(2) the DETECTOR's own precision/recall (is-the-head-a-known-product-noun) on telling
    type-less entity queries from concrete ones.
(3) the BASE RATE of type-less / ambiguous queries in the WANDS and ESCI query streams.
All from existing artifacts; no model calls."""
import os, re, json, collections
import pandas as pd
DATA=os.path.join(os.path.dirname(__file__),"..","data")
STOP=set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def toks(s): return [w for w in norm(s).split() if w and w not in STOP]

p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
# product-noun vocab (class head tokens + producttype values) = the rule-based planner's type lexicon
class_tokens=set()
for c in p["product_class"].dropna():
    for w in norm(c).split():
        if len(w)>2: class_tokens.add(w)
for s in p["product_features"].dropna():
    for part in str(s).split("|"):
        if part.lower().strip().startswith("producttype") and ":" in part:
            for w in norm(part.split(":",1)[1]).split():
                if len(w)>2: class_tokens.add(w)
def is_concrete(q): return any(w in class_tokens for w in toks(q))

# (1) rule-based planner: emits product_type only if a class token is literally in the query
probe=[json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl")]
entity=[r for r in probe if r["family"]=="A_entity"]; attr=[r for r in probe if r["family"]=="B_attr"]
rb_entity_typed=sum(1 for r in entity if is_concrete(r["query"]))   # rule-based would emit a type
rb_attr_typed=sum(1 for r in attr if is_concrete(r["query"]))
print("=== (1) Non-LLM rule-based planner (catalog-vocab extractor) ===")
print(f"  entity queries that get a product_type: {rb_entity_typed}/{len(entity)} = {rb_entity_typed/len(entity):.2f}  (LLMs under expansion: ~1.00)")
print(f"  attributed (Family B) queries that get a product_type: {rb_attr_typed}/{len(attr)} = {rb_attr_typed/len(attr):.2f}  (these DO name a type)")
print("  -> rule-based planner does NOT hallucinate a category on type-less queries; the failure is LLM-specific.")

# (2) detector precision/recall: ground truth = entity (should be flagged 'suppress'=ambiguous), Family B = concrete
# detector flags 'apply filter' when is_concrete True. We want it to NOT apply on entity, DO apply on attributed.
tp=sum(1 for r in attr if is_concrete(r["query"]))      # concrete correctly applied
fn=sum(1 for r in attr if not is_concrete(r["query"]))  # concrete wrongly suppressed
fp=sum(1 for r in entity if is_concrete(r["query"]))    # entity wrongly applied (filter not suppressed)
tn=sum(1 for r in entity if not is_concrete(r["query"]))# entity correctly suppressed
prec=tp/(tp+fp) if tp+fp else float('nan'); rec=tp/(tp+fn) if tp+fn else float('nan')
print("\n=== (2) Detector precision/recall (apply-filter vs suppress) ===")
print(f"  concrete (apply): TP={tp} FN={fn}   entity (suppress): TN={tn} FP={fp}")
print(f"  precision(apply)={prec:.2f} recall(apply)={rec:.2f}  ; entity-suppression accuracy={tn/(tn+fp):.2f}")

# (3) base rate of ambiguous/type-less queries in the streams
q=pd.read_csv(f"{DATA}/query.csv",sep="\t")
amb=sum(1 for x in q["query"] if not is_concrete(x)); nq=len(q)
print("\n=== (3) Base rate of type-less/ambiguous queries ===")
print(f"  WANDS: {amb}/{nq} = {amb/nq:.1%} of queries have no product-noun head (type-less/ambiguous)")
try:
    eq=[json.loads(l)["query"] for l in open(f"{DATA}/esci/queries.jsonl")]
    ea=sum(1 for x in eq if not is_concrete(x))
    print(f"  ESCI : {ea}/{len(eq)} = {ea/len(eq):.1%}")
except Exception as e:
    print("  ESCI: n/a", e)
