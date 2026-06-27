#!/usr/bin/env python
"""Regenerate the data-driven figures from results/ metrics.
Currently regenerates Fig. 2 (universality, two-domain forced-rate bars) and
Fig. 3 (slot-locus). Fig. 1 is a hand-drawn pipeline schematic; Figs. 4-6
(enforcement sweep, ablation, dose-response) are provided as PNGs in results/.
Usage: python scripts/make_figs.py"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES = os.path.join(os.path.dirname(__file__), "..", "results")
def L(p):
    try: return json.load(open(os.path.join(RES, p)))
    except Exception: return None
MAIN = [("Qwen2.5-7B","qwen4"),("Llama-3.1-8B","llama4"),("Mistral-7B","mistral4"),("Gemma-2-9B","gemma4"),
        ("Haiku 4.5","haiku"),("Sonnet 4.6","sonnet"),("Llama-3.3-70B","together-llama70"),("gpt-4o","gpt4o"),("gpt-5.2","gpt52")]

# --- Fig 2: universality, forced product_type rate, conservative/expansion x WANDS/Amazon ---
names=[n for n,_ in MAIN]
wc=[L(f"probe-{t}-con__probe_metrics.json")["A_product_type_forced_rate"][0] for _,t in MAIN]
we=[L(f"probe-{t}-exp__probe_metrics.json")["A_product_type_forced_rate"][0] for _,t in MAIN]
ac=[L(f"probe-esci-{t}-con__probe_metrics.json")["A_product_type_forced_rate"][0] for _,t in MAIN]
ae=[L(f"probe-esci-{t}-exp__probe_metrics.json")["A_product_type_forced_rate"][0] for _,t in MAIN]
x=np.arange(len(MAIN)); w=0.2
fig,ax=plt.subplots(figsize=(11,4.4))
ax.bar(x-1.5*w,wc,w,label="WANDS conservative",color="#c6dbef")
ax.bar(x-0.5*w,we,w,label="WANDS expansion",color="#08519c")
ax.bar(x+0.5*w,ac,w,label="Amazon conservative",color="#fdd0a2")
ax.bar(x+1.5*w,ae,w,label="Amazon expansion",color="#a63603")
ax.set_xticks(x); ax.set_xticklabels(names,rotation=30,ha="right",fontsize=8)
ax.set_ylabel("forced product_type rate\n(type-less entity queries)"); ax.set_ylim(0,1.08)
ax.axhline(1.0,ls=":",c="grey",lw=0.7); ax.legend(loc="lower center",ncol=2,fontsize=8)
ax.set_title("Under expansion every model routes into the category slot ~100% on BOTH catalogs (9 models, 6 providers)",fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(RES,"fig2_universality.png"),dpi=150,bbox_inches="tight"); plt.close()
print("wrote fig2_universality.png")

# --- Fig 3: slot-locus, P(spurious injection excludes a relevant item) per slot ---
slots=["product_type","color","material","style"]; agg={s:[] for s in slots}
for _,t in MAIN:
    m=L(f"{t}-exp__metrics_v2.json")
    if m and "per_slot" in m:
        for s in slots:
            d=m["per_slot"].get(s)
            if d and d["spurious"]>0: agg[s].append(d["harmful"]/d["spurious"])
means=[float(np.mean(agg[s])) for s in slots]
fig,ax=plt.subplots(figsize=(5.5,4))
ax.bar(["category\n(product_type)","color","material","style"],means,color=["#a50f15","#fdae6b","#fdae6b","#fdae6b"])
for i,v in enumerate(means): ax.text(i,v+0.01,f"{v:.2f}",ha="center",fontsize=9)
ax.set_ylabel("P(spurious injection excludes a relevant item)"); ax.set_ylim(0,1.05)
ax.set_title("A spurious category filter almost always excludes a relevant item\n(near-universal facet coverage); soft slots far less",fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES,"fig3_slotlocus.png"),dpi=150,bbox_inches="tight"); plt.close()
print("wrote fig3_slotlocus.png  slot-locus:", {s:round(float(np.mean(agg[s])),2) for s in slots})
