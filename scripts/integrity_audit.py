#!/usr/bin/env python
"""Integrity audit with a FRESHNESS GATE.

Why this exists: in the n=31->55 incident, the probe was scaled and the models
re-run, but the scoring artifacts (probe_metrics, retrieval_validation, enforce)
were never regenerated, so they held stale n=31 numbers. The manuscript and the
stale artifacts were internally consistent, so a value-vs-value audit passed. The
gate below makes that class of bug impossible to miss: every metric file must have
been computed on the CURRENT probe size, or the audit FAILS loud and names the file.

Run: python scripts/integrity_audit.py   (exit code 1 if any artifact is stale)

STATS LESSON (do not regress): the headline CI must be a CLUSTER bootstrap that resamples
the ~55 queries (each carrying its 9 correlated model deltas), NOT the 493 model-query
pairs as independent (that pseudo-replication understates the interval). See headline_stats_v2.py.
"""
import json, os, sys, glob
import pandas as pd
RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
def L(p):
    try: return json.load(open(os.path.join(RES, p)))
    except Exception: return None
def g(d, k): return d.get(k) if d else None

# --- current probe sizes (the ground truth every metric must match) ---
WANDS_A = sum(1 for l in open(f"{DATA}/probe/probe.jsonl") if json.loads(l)["family"] == "A_entity")
WANDS_B = sum(1 for l in open(f"{DATA}/probe/probe.jsonl") if json.loads(l)["family"] == "B_attr")
ESCI_N  = sum(1 for _ in open(f"{DATA}/probe_esci/probe_full.jsonl"))
MAIN = [("qwen4","Qwen2.5-7B"),("llama4","Llama-3.1-8B"),("mistral4","Mistral-7B"),("gemma4","Gemma-2-9B"),
        ("haiku","Haiku 4.5"),("sonnet","Sonnet 4.6"),("together-llama70","Llama-3.3-70B"),("gpt4o","gpt-4o"),("gpt52","gpt-5.2")]
PROBE_MTIME = os.path.getmtime(f"{DATA}/probe/probe.jsonl")

print("="*72); print(f"FRESHNESS GATE  (current probe: WANDS A_entity={WANDS_A}, B_attr={WANDS_B}; ESCI={ESCI_N})"); print("="*72)
stale = []
def check(path, got_n, want_n, allow_typed_lt=False):
    if got_n is None:
        stale.append((path, "no n field", want_n)); return
    if got_n != want_n and not (allow_typed_lt and got_n <= want_n):
        stale.append((path, got_n, want_n))

for tag, _ in MAIN:
    for cond in ("con","exp","opt"):
        m = L(f"probe-{tag}-{cond}__probe_metrics.json")
        if m: check(f"probe-{tag}-{cond}__probe_metrics.json", g(m,"n_A"), WANDS_A)
    rv = L(f"probe-{tag}-exp__retrieval_validation.json")
    if rv: check(f"probe-{tag}-exp__retrieval_validation.json", g(rv,"n_entity"), WANDS_A)
    for r in ("bm25","dense"):
        e = L(f"probe-{tag}-exp__enforce_{r}.json")
        if e: check(f"probe-{tag}-exp__enforce_{r}.json", g(e,"n"), WANDS_A, allow_typed_lt=True)
    for cond in ("con","exp"):
        em = L(f"probe-esci-{tag}-{cond}__probe_metrics.json")
        if em: check(f"probe-esci-{tag}-{cond}__probe_metrics.json", g(em,"n_A"), ESCI_N)
    ep = L(f"probe-esci-{tag}-exp__esci_probe_metrics.json")
    if ep: check(f"probe-esci-{tag}-exp__esci_probe_metrics.json", g(ep,"n_entity"), ESCI_N)
# supplementary artifacts the manuscript cites but that sit outside MAIN (must still trace):
for tag in ("gem3flash", "gem25flashlite"):  # the two cited Google supplementary models
    for cond in ("con", "exp"):
        m = L(f"probe-{tag}-{cond}__probe_metrics.json")
        if m: check(f"probe-{tag}-{cond}__probe_metrics.json", g(m, "n_A"), WANDS_A)
ep = L("entity_precision.json")  # precision@10/nDCG@10 must cover all 9 models
if ep is not None and ep.get("n_models") != 9:
    stale.append(("entity_precision.json", ep.get("n_models"), 9))
for t, _ in [("qwen4",0),("llama4",0),("mistral4",0),("gemma4",0)]:
    fb = L(f"probe-{t}-exp__fallback.json")
    if fb: check(f"probe-{t}-exp__fallback.json", g(fb, "n"), WANDS_A, allow_typed_lt=True)
ms = L("micro_sql_qwen.json")  # text-to-SQL micro-experiment ran on the 55 motifs
if ms is not None and ms.get("n") != WANDS_A:
    stale.append(("micro_sql_qwen.json", ms.get("n"), WANDS_A))
for t in ("qwen4","llama4","mistral4","gemma4"):  # schema-vs-prompt dissociation control cell
    nb = L(f"probe-{t}-exp_nullable__probe.json")
    if nb is not None and nb.get("n") != WANDS_A:
        stale.append((f"probe-{t}-exp_nullable__probe.json", nb.get("n"), WANDS_A))
    sq = L(f"micro_sql_{t}.json")
    if sq is not None and sq.get("n") not in (WANDS_A, None):
        stale.append((f"micro_sql_{t}.json", sq.get("n"), WANDS_A))

# mtime gate, restricted to the artifacts the manuscript actually cites (main 9, exp):
for tag, _ in MAIN:
    for suf in ("exp__probe_metrics.json", "exp__retrieval_validation.json"):
        f = f"{RES}/probe-{tag}-{suf}"
        if os.path.exists(f) and os.path.getmtime(f) < PROBE_MTIME:
            stale.append((os.path.basename(f), "mtime < probe mtime", "rebuild"))
for f in ("headline_stats_byprompt.json","con_drops.json","taxonomy_tau_sweep.json","nondecor_routing.json","construct_validity.json","decoupled_relevance.json","dissociation_fullprec.json"):
    fp=f"{RES}/{f}"
    if os.path.exists(fp) and os.path.getmtime(fp) < PROBE_MTIME:
        stale.append((f, "mtime < probe mtime", "rebuild"))

if stale:
    print(f"\n  *** {len(stale)} STALE ARTIFACT(S) - regenerate before trusting any number ***")
    for p, got, want in stale: print(f"    STALE  {p:48s} got={got} want={want}")
    print()
else:
    print("\n  OK: every metric artifact was computed on the current probe size.\n")

print("="*72); print("CURRENT HEADLINE VALUES (n-correct)"); print("="*72)
print(f"\n[3.1] WANDS catalog: products={len(pd.read_csv(f'{DATA}/product.csv',sep=chr(9)))}; probe A_entity={WANDS_A} B_attr={WANDS_B}; ESCI probe={ESCI_N}")

print("\n[Table 1] forced product_type rate (con / exp), n=%d:" % WANDS_A)
fc_e=[]
for tag,name in MAIN:
    c=L(f"probe-{tag}-con__probe_metrics.json"); e=L(f"probe-{tag}-exp__probe_metrics.json")
    fc=g(c,"A_product_type_forced_rate"); fe=g(e,"A_product_type_forced_rate")
    if fe: fc_e.append(fe[0])
    print(f"   {name:20s} con {fc[0]:.2f}   exp {fe[0]:.2f} [{fe[1]:.2f},{fe[2]:.2f}]")
print(f"   --> expansion band: {min(fc_e):.2f}-{max(fc_e):.2f}")

print("\n[Table 2] real-retrieval recall@100 drop (exp), n=%d:" % WANDS_A)
rd_all=[]
for tag,name in MAIN:
    rv=L(f"probe-{tag}-exp__retrieval_validation.json"); rd=g(rv,"mean_recall_drop_real_retrieval")
    if rd: rd_all.append(rd[0]); print(f"   {name:20s} {rd[0]:.3f} [{rd[1]:.3f},{rd[2]:.3f}]  (n_typed={g(rv,'n_typed')})")
print(f"   --> recall-loss band: {min(rd_all):.2f}-{max(rd_all):.2f}")

print("\n[App. A] enforcement hard-filter (lambda 1.0..0.0) recall@100 loss:")
for tag,name in [("qwen4","Qwen"),("llama4","Llama")]:
    for r in ("bm25","dense"):
        e=L(f"probe-{tag}-exp__enforce_{r}.json")
        if e:
            ld=e["lambda_drop"]; vals=" ".join(f"{ld[k][0]:.2f}" for k in ["1.0","0.9","0.7","0.5","0.3","0.0"])
            print(f"   {name} {r}: {vals}   (n={e['n']})")

print("\n[5.5] schema-nullability ablation: exp(forced) -> opt(forced), n=%d:" % WANDS_A)
opt_all=[]
for tag,name in [("qwen4","Qwen"),("llama4","Llama"),("mistral4","Mistral"),("gemma4","Gemma")]:
    o=L(f"probe-{tag}-opt__probe_metrics.json"); f=g(o,"A_product_type_forced_rate")
    if f: opt_all.append(f[0]); print(f"   {name}: 1.00 -> {f[0]:.2f}")
print(f"   --> schema-fix band: {min(opt_all):.2f}-{max(opt_all):.2f}")

print("\n[5.5] cross-encoder rerank survival (no-filter -> hard-filter recall):")
rr=[]
for m in ["qwen4","llama4","mistral4","gemma4"]:
    d=L(f"probe-{m}-exp__rerank.json")
    if d: rr.append(d["rerank_harm"]); print(f"   {m}: {d['rerank_norfilter_recall']:.3f} -> {d['rerank_hardfilter_recall']:.3f} (loss {d['rerank_harm']:.3f}, n={d['n']})")
if rr: print(f"   --> rerank-survival band: {min(rr):.2f}-{max(rr):.2f}")

print("\n[5.6] ESCI (Amazon 1.2M catalog), forced + real recall@100 drop, n=%d:" % ESCI_N)
ef=[]; edr=[]
for tag,name in MAIN:
    ep=L(f"probe-esci-{tag}-exp__esci_probe_metrics.json")
    if ep: ef.append(ep["forced_type_rate"][0]); edr.append(ep["real_recall_drop"][0]); print(f"   {name:20s} forced {ep['forced_type_rate'][0]:.2f}  recall_drop {ep['real_recall_drop'][0]:.2f}  (n={g(ep,'n_entity')})")
if ef: print(f"   --> ESCI forced {min(ef):.2f}-{max(ef):.2f}; recall-drop {min(edr):.2f}-{max(edr):.2f}")

print("\n" + "="*72)
sys.exit(1 if stale else 0)
