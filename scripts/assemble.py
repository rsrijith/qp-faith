#!/usr/bin/env python
"""Assemble the dose-response curve (SCR vs prompt-aggressiveness, per model) from
the v2 metrics, save a plot, and print a consolidated table."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
LEVELS = [("con", "conservative"), ("imp", "imply"), ("ast", "assist"),
          ("enr", "enrich"), ("exp", "expansion")]
MODELS = ["qwen4", "llama4", "mistral4", "gemma4"]

def L(p):
    try: return json.load(open(p))
    except Exception: return None

def main():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(LEVELS))
    table = {}
    for m in MODELS:
        ys, los, his = [], [], []
        for short, _ in LEVELS:
            d = L(f"{RES}/{m}-{short}__metrics_v2.json")
            if d and d.get("frac_q_with_spurious"):
                pt, lo, hi = d["frac_q_with_spurious"]
            else:
                pt = lo = hi = np.nan
            ys.append(pt); los.append(lo); his.append(hi)
        table[m] = ys
        ax.errorbar(x, ys, yerr=[np.array(ys)-np.array(los), np.array(his)-np.array(ys)],
                    marker="o", capsize=3, label=m.replace("4", ""))
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in LEVELS], rotation=20)
    ax.set_ylabel("frac. queries with >=1 spurious constraint")
    ax.set_xlabel("prompt inference-aggressiveness")
    ax.set_title("Dose-response: spurious injection vs prompt aggressiveness")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{RES}/dose_response.png", dpi=140)
    json.dump({"levels": [n for _, n in LEVELS], "scr_by_model": table},
              open(f"{RES}/dose_response.json", "w"), indent=2, default=float)
    print("DOSE-RESPONSE SCR (frac queries with spurious):")
    print("model     " + "  ".join(f"{n[:6]:>6s}" for _, n in LEVELS))
    mono = True
    for m in MODELS:
        ys = table[m]
        print(f"{m:9s} " + "  ".join(f"{v:6.2f}" if v==v else "   nan" for v in ys))
        clean = [v for v in ys if v == v]
        if any(clean[i+1] < clean[i] - 0.02 for i in range(len(clean)-1)): mono = False
    print(f"\nmonotone non-decreasing (within 0.02 tol): {mono}")
    print(f"saved {RES}/dose_response.png and dose_response.json")

if __name__ == "__main__":
    main()
