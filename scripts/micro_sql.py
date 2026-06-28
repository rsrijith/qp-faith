#!/usr/bin/env python
"""P3 micro-experiment: does the over-routing generalize beyond e-commerce category
filters to text-to-SQL? Same type-less entity motifs, but the planner must emit a SQL
WHERE clause over a products table. We measure how often it commits a spurious category/
type equality predicate (the SQL analogue of the product_type slot). One demonstrated
instance of the conjecture in Section 6, not a full study."""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(__file__))
from run_pilot import load_planner
DATA=os.path.join(os.path.dirname(__file__),"..","data"); RES=os.path.join(os.path.dirname(__file__),"..","results")
SQL_PROMPT=("You query a products database with schema: "
"products(id, title, description, category, color, material, brand, price). "
"The shopper searched: '{q}'. Write ONE SQL WHERE clause that narrows the catalog as much "
"as possible to the most relevant products; infer as many constraints as you reasonably can. "
"Output ONLY the WHERE clause (starting with WHERE), nothing else.")
motifs=[json.loads(l)["query"] for l in open(f"{DATA}/probe/probe.jsonl") if json.loads(l)["family"]=="A_entity"]
planner=load_planner("mlx-community/Qwen2.5-7B-Instruct-4bit", SQL_PROMPT)
cat_pred=re.compile(r"\b(category|product_type|type)\s*(=|like|in)\b", re.I)
eq_pred=re.compile(r"\b(category|product_type|type)\s*=\s*'([^']*)'", re.I)
rows=[]; n_cat=0; n_overconstrain=0
for q in motifs:
    raw=planner(q)
    where=raw[raw.upper().find("WHERE"):] if "WHERE" in raw.upper() else raw
    where=where.split("\n")[0][:300]
    has_cat=bool(cat_pred.search(where))
    eqs=eq_pred.findall(where); spurious_eq=any(q.lower() not in v.lower() and v.lower() not in q.lower() for _,v in eqs)
    n_preds=len(re.findall(r"\b(=|like|in)\b", where, re.I))
    n_cat+=has_cat; n_overconstrain+= (n_preds>=2)
    rows_eq=locals().get("rows_eq",0)
    rows.append({"q":q,"where":where,"category_predicate":has_cat,"spurious_category_equality":spurious_eq,"n_predicates":n_preds})
N=len(motifs)
n_spur_eq=sum(1 for r in rows if r.get("spurious_category_equality"))
rep={"model":"Qwen2.5-7B-Instruct-4bit","n":N,
     "category_equality_spurious_rate":round(n_spur_eq/N,3),
     "any_category_predicate_rate_incl_faithful_LIKE":round(n_cat/N,3),
     "multi_predicate_rate":round(n_overconstrain/N,3),"rows":rows}
json.dump(rep,open(f"{RES}/micro_sql_qwen.json","w"),indent=2)
print(f"N={N} type-less SQL queries")
print(f"  emits a spurious category/type WHERE predicate: {n_cat}/{N} = {n_cat/N:.0%}")
print(f"  emits >=2 predicates (over-constrains): {n_overconstrain}/{N} = {n_overconstrain/N:.0%}")
print("  examples:")
for r in rows[:5]: print(f"    '{r['q']}' -> {r['where'][:80]}")
