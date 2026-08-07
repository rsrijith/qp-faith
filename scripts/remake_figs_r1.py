#!/usr/bin/env python
"""R1 revision: regenerate Fig 2 (readable, no rotated labels; R1 minor 2) and
Fig 3 (grouped bar, full-hard vs conflict-only per slot; R2 minor 7) as vector
PDFs written straight into submission_ipm/. Data come from results/ JSONs.
Usage: ./.venv/bin/python scripts/remake_figs_r1.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "submission_ipm")

def L(p):
    return json.load(open(os.path.join(RES, p)))

MAIN = [("Qwen2.5-7B", "qwen4"), ("Llama-3.1-8B", "llama4"), ("Mistral-7B", "mistral4"),
        ("Gemma-2-9B", "gemma4"), ("Haiku 4.5", "haiku"), ("Sonnet 4.6", "sonnet"),
        ("Llama-3.3-70B", "together-llama70"), ("gpt-4o", "gpt4o"), ("gpt-5.2", "gpt52")]

# ---------- Fig 2: two horizontal-bar panels (WANDS | Amazon), labels un-rotated ----------
names = [n for n, _ in MAIN]
def rate(tag):
    return L(f"{tag}__probe_metrics.json")["A_product_type_forced_rate"][0]
wc = [rate(f"probe-{t}-con") for _, t in MAIN]
we = [rate(f"probe-{t}-exp") for _, t in MAIN]
ac = [rate(f"probe-esci-{t}-con") for _, t in MAIN]
ae = [rate(f"probe-esci-{t}-exp") for _, t in MAIN]

y = np.arange(len(MAIN))[::-1]  # first model on top
h = 0.38
fig, (axW, axA) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, con, exp, title in [(axW, wc, we, "WANDS (home goods)"),
                            (axA, ac, ae, "Amazon (full 1.2M catalog)")]:
    ax.barh(y + h/2, con, h, label="conservative prompt", color="#c6dbef", edgecolor="#6699cc", linewidth=0.4)
    ax.barh(y - h/2, exp, h, label="expansion prompt", color="#08519c")
    ax.set_xlim(0, 1.05)
    ax.axvline(1.0, ls=":", c="grey", lw=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("forced product_type rate (type-less queries)", fontsize=9)
    ax.tick_params(labelsize=8)
axW.set_yticks(y)
axW.set_yticklabels(names, fontsize=8.5)
axA.legend(loc="lower right", fontsize=8, framealpha=0.9)
fig.suptitle("Under expansion every planner routes into the category slot ~100% on both catalogs; "
             "conservative behavior varies widely and does not track capability",
             fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_universality.pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote fig2_universality.pdf")

# ---------- Fig 3: grouped bar, full-hard vs conflict-only per slot (from the matrix) ----------
M = L("slot_enforcement_matrix.json")["matrix"]
slots = ["category", "color", "material", "style"]  # brand is degenerate on WANDS
labels = ["category\n(product_class)", "color", "material", "style"]
fh = np.array([M[s]["full_hard_filter_exclusion"][0] for s in slots])
fh_lo = np.array([M[s]["full_hard_filter_exclusion"][1] for s in slots])
fh_hi = np.array([M[s]["full_hard_filter_exclusion"][2] for s in slots])
co = np.array([M[s]["conflict_only_exclusion"][0] for s in slots])
co_lo = np.array([M[s]["conflict_only_exclusion"][1] for s in slots])
co_hi = np.array([M[s]["conflict_only_exclusion"][2] for s in slots])

x = np.arange(len(slots))
w = 0.36
fig, ax = plt.subplots(figsize=(6.4, 4.0))
b1 = ax.bar(x - w/2, fh, w, yerr=[fh - fh_lo, fh_hi - fh], capsize=3,
            label="full hard-filter (missing facets excluded)", color="#a50f15")
b2 = ax.bar(x + w/2, co, w, yerr=[co - co_lo, co_hi - co], capsize=3,
            label="conflict-only (missing facets retained)", color="#fdae6b", edgecolor="#d9822b", linewidth=0.4)
for bars, vals in [(b1, fh), (b2, co)]:
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.08)
ax.set_ylabel("P(spurious injection excludes a relevant product)", fontsize=9)
ax.axhline(0, color="black", lw=0.6)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.set_title("Under full hard-filter every slot excludes (a coverage effect);\n"
             "only category stays high under conflict-only (the mechanism: enforcement x spread)",
             fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_slotlocus.pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote fig3_slotlocus.pdf  full-hard:", dict(zip(slots, fh.round(3))),
      " conflict-only:", dict(zip(slots, co.round(3))))
