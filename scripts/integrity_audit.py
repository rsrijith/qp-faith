#!/usr/bin/env python
"""Integrity audit: print the ground-truth value (from persisted results/ artifacts)
for every numeric claim in PAPER.md, so each can be checked against the manuscript.
Read the output next to the paper; any mismatch is a must-fix."""
import json, os, re, collections, glob
RES = os.path.join(os.path.dirname(__file__), "..", "results")
def L(p):
    try: return json.load(open(os.path.join(RES, p)))
    except Exception as e: return None
def g(d, k):
    return d.get(k) if d else None

print("="*70); print("INTEGRITY AUDIT - paper claim vs persisted artifact"); print("="*70)

# --- Catalog facts (3.1) ---
import pandas as pd
DATA=os.path.join(os.path.dirname(__file__),"..","data")
p=pd.read_csv(f"{DATA}/product.csv",sep="\t"); q=pd.read_csv(f"{DATA}/query.csv",sep="\t"); lab=pd.read_csv(f"{DATA}/label.csv",sep="\t")
print(f"\n[3.1] WANDS: queries={len(q)} (paper 480) products={len(p)} (paper 42,994) pairs={len(lab)} (paper 233,448)")

# --- Probe scale (3.4) ---
probe=[json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl")]
nA=sum(1 for r in probe if r['family']=='A_entity'); nB=sum(1 for r in probe if r['family']=='B_attr')
print(f"[3.4] WANDS probe: A_entity={nA} (paper 43) B_attr={nB} (paper 18)")
try:
    pe=[json.loads(l) for l in open(f"{DATA}/probe_esci/probe.jsonl")]; print(f"[3.4] ESCI probe entity={len(pe)} (paper ~60)")
except: print("[3.4] ESCI probe: missing")

# --- Reference coverage (3.5) ---
gm=L("../data/gold/manifest.json") or json.load(open(f"{DATA}/gold/manifest.json"))
print(f"[3.5] reference coverage: {gm.get('coverage')} (paper pt87/color49/material24/style18)")

# --- Table 1 + Table 2: routing + forced rate + recall drop ---
print("\n[Table 1/2] per-model forced-category rate (con/exp) and real recall@100 drop (exp):")
models=[("qwen4","Qwen2.5-7B"),("llama4","Llama-3.1-8B"),("mistral4","Mistral-7B"),("gemma4","Gemma-2-9B"),
        ("haiku","Haiku 4.5"),("sonnet","Sonnet 4.6"),("together-llama70","Llama-3.3-70B"),("gpt4o","gpt-4o"),("gpt52","gpt-5.2")]
for tag,name in models:
    pc=L(f"probe-{tag}-con__probe_metrics.json"); pe2=L(f"probe-{tag}-exp__probe_metrics.json")
    rv=L(f"probe-{tag}-exp__retrieval_validation.json")
    fc=g(pc,"A_product_type_forced_rate"); fe=g(pe2,"A_product_type_forced_rate")
    rd=g(rv,"mean_recall_drop_real_retrieval")
    fcs=f"{fc[0]:.2f}" if fc else "--"; fes=f"{fe[0]:.2f}" if fe else "--"; rds=f"{rd[0]:.2f}" if rd else "--"
    print(f"   {name:22s} con {fcs}  exp {fes}  recall_drop {rds}")

# --- slot routing soft-only (Table 1) for flash vs a collapser ---
def soft_only(tag):
    rows=[r for r in (json.loads(l) for l in open(f"{RES}/probe-{tag}-exp__probe_plans.jsonl")) if r["family"]=="A_entity"]
    n=len(rows); pt=sum(1 for r in rows if "product_type" in r["plan"])
    so=sum(1 for r in rows if ("style" in r["plan"] or "color" in r["plan"] or "material" in r["plan"]) and "product_type" not in r["plan"])
    return f"category {pt/n:.0%} soft-only {so/n:.0%}"
print(f"\n[Table 1] Gemini 3 Flash: {soft_only('gem3flash-api')} (now collapses, no soft-only exception)")
print(f"[Table 1] Qwen: {soft_only('qwen4')} (paper 100%/0%)")

# --- Multi-class control (5.2) ---
print("\n[5.2] multi-class control (clean/artifact, organic exp):")
for tag,name in [("qwen4","Qwen"),("llama4","Llama"),("mistral4","Mistral"),("gemma4","Gemma")]:
    m=L(f"{tag}-exp__metrics_v2.json")
    print(f"   {name}: clean {g(m,'product_type_clean_harm_singleclass')} artifact {g(m,'product_type_harm_multiclass_artifact')} (paper 9/35,43/94,80/125,71/129)")

# --- Enforcement (Appendix A) ---
print("\n[Appendix A] enforcement curve (lambda 1.0..0.0):")
for tag in ["probe-qwen4-exp","probe-llama4-exp"]:
    for r in ["bm25","dense"]:
        e=L(f"{tag}__enforce_{r}.json")
        if e:
            ld=e["lambda_drop"]; vals=" ".join(f"{ld[k][0]:.2f}" for k in ["1.0","0.9","0.7","0.5","0.3","0.0"])
            print(f"   {tag} {r}: {vals}")

# --- Schema ablation (5.5) ---
print("\n[5.5] schema-nullability ablation (entity forced rate after fix):")
for tag,name in [("qwen4","Qwen"),("llama4","Llama"),("mistral4","Mistral"),("gemma4","Gemma")]:
    m=L(f"probe-{tag}-opt__probe_metrics.json")
    f=g(m,"A_product_type_forced_rate")
    print(f"   {name}: {f[0]:.2f}" if f else f"   {name}: missing", " (paper .16/.13/.19/.23)")

# --- Recovery (5.5) ---
print("\n[5.5] detector recovery (recovery.json):")
for tag,name in [("qwen4","Qwen"),("llama4","Llama"),("mistral4","Mistral"),("gemma4","Gemma")]:
    rc=L(f"{tag}-exp__recovery.json")
    if rc:
        a=rc.get("ambiguous",{}); c=rc.get("concrete",{})
        amb_f=a.get("recall",{}).get("FILTER",[None])[0]; amb_g=a.get("recall",{}).get("GATED",[None])[0]
        con_n=c.get("ndcg",{}).get("NOFILTER",[None])[0]; con_f=c.get("ndcg",{}).get("FILTER",[None])[0]
        rec=(amb_g-amb_f) if (amb_f is not None and amb_g is not None) else None
        ben=(con_f-con_n) if (con_n is not None and con_f is not None) else None
        print(f"   {name}: amb recall recovered {rec:+.3f}  concrete filter benefit {ben:+.3f}" if rec is not None else f"   {name}: parse issue")
    else: print(f"   {name}: recovery.json missing")

# --- ESCI (5.6) ---
print("\n[5.6] ESCI second domain:")
for tag,name in [("qwen4","Qwen"),("llama4","Llama"),("mistral4","Mistral"),("gemma4","Gemma")]:
    ep=L(f"probe-esci-{tag}-exp__esci_probe_metrics.json")
    if ep: print(f"   {name} esci-probe: forced {ep['forced_type_rate'][0]:.2f} recall_drop {ep['real_recall_drop'][0]:.2f}")
    ec=L(f"esci-{tag}-con__metrics.json"); ee=L(f"esci-{tag}-exp__metrics.json")
    if ec and ee: print(f"        organic brand/color frac-q-spurious con {ec['frac_q_spurious'][0]:.2f} exp {ee['frac_q_spurious'][0]:.2f}")

# --- taxonomy (5.4) ---
tags=["probe-qwen4-exp","probe-llama4-exp","probe-mistral4-exp","probe-gemma4-exp","probe-haiku-exp","probe-sonnet-exp","probe-together-llama70-exp","probe-gpt4o-exp","probe-gpt52-exp"]
vals=collections.Counter(); n=0
for t in tags:
    try: rows=[json.loads(l) for l in open(f"{RES}/{t}__probe_plans.jsonl")]
    except: continue
    for r in rows:
        if r["family"]=="A_entity" and r["plan"].get("product_type"):
            vals[r["plan"]["product_type"].strip().lower()]+=1; n+=1
generic={"decor","home decor","decoration","furniture","wall decor","wall art","art","outdoor decor","ornament","figurine","decorative figurine"}
gg=sum(c for v,c in vals.items() if v in generic)
print(f"\n[5.4] taxonomy: total injected={n} (paper 493) generic={gg/n:.0%} (paper four patterns) top={vals.most_common(6)}")
print("\n" + "="*70)

# --- human relevance eval (3.4) ---
he=L("human_eval_summary.json")
if he:
    a1=he["a1_namematch_relevant"]; a2=he["a2_namematch_relevant"]
    print(f"\n[3.4] human eval ({he['annotators']} annotators): A1 namematch {a1[0]}/{a1[1]}={a1[0]/a1[1]:.0%} (paper 98%); "
          f"A2 {a2[0]}/{a2[1]}={a2[0]/a2[1]:.0%} (paper 92%); "
          f"Cohen kappa {he['cohen_kappa']} (paper 0.72), raw agreement {he['raw_agreement']:.0%}")

# [5.5] cross-encoder rerank: harm surviving a reranking stage
print("\n[5.5] rerank (no-filter -> hard-filter recall@100, loss survives reranker):")
import json as _j
for m in ["qwen4","llama4","mistral4","gemma4"]:
    try:
        d=_j.load(open(f"{RES}/probe-{m}-exp__rerank.json"))
        print(f"   {m}: {d['rerank_norfilter_recall']:.3f} -> {d['rerank_hardfilter_recall']:.3f} (loss {d['rerank_harm']:.3f})  [paper 0.20-0.43]")
    except Exception as e: print(f"   {m}: {e}")
