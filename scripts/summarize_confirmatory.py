#!/usr/bin/env python
"""Summarize the confirmatory electronics/apparel runs: per-cell forced-type rate and
recall@100 drop, plus a per-domain pooled expansion harm (mean over the 4 local models
with a model-level bootstrap CI). Reads results/dom-<dom>-<model>-<mode>__esci_probe_metrics.json."""
import os, json, glob
import numpy as np
RES = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = ["qwen4", "llama4", "mistral4", "gemma4"]
DOMS = ["electronics", "apparel"]

def load(tag):
    p = f"{RES}/{tag}__esci_probe_metrics.json"
    if not os.path.exists(p): return None
    return json.load(open(p))

def boot_mean(vals, n=5000, seed=13):
    vals = np.array([v for v in vals if v == v], float)
    if not len(vals): return (float("nan"),)*3
    rng = np.random.default_rng(seed)
    s = vals[rng.integers(0, len(vals), (n, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

print(f"{'domain':12} {'model':9} {'mode':12} {'n':>4} {'forced':>16} {'recall_drop':>18}")
print("-"*75)
rows = {}
for dom in DOMS:
    for m in MODELS:
        for mode in ["conservative", "expansion"]:
            r = load(f"dom-{dom}-{m}-{mode}")
            if not r:
                print(f"{dom:12} {m:9} {mode:12}  MISSING")
                continue
            ft = r["forced_type_rate"]; rd = r["real_recall_drop"]
            rows[(dom, m, mode)] = (r["n_entity"], ft, rd)
            print(f"{dom:12} {m:9} {mode:12} {r['n_entity']:>4} "
                  f"{ft[0]:.2f}[{ft[1]:.2f},{ft[2]:.2f}]   {rd[0]:.2f}[{rd[1]:.2f},{rd[2]:.2f}]")

print("\n=== per-domain pooled expansion harm (mean over 4 models, model-level bootstrap) ===")
for dom in DOMS:
    exp_drops = [rows[(dom, m, "expansion")][2][0] for m in MODELS if (dom, m, "expansion") in rows]
    con_drops = [rows[(dom, m, "conservative")][2][0] for m in MODELS if (dom, m, "conservative") in rows]
    exp_forced = [rows[(dom, m, "expansion")][1][0] for m in MODELS if (dom, m, "expansion") in rows]
    if exp_drops:
        b = boot_mean(exp_drops)
        print(f"{dom:12} expansion harm  {b[0]:.2f} [{b[1]:.2f},{b[2]:.2f}]   "
              f"(per-model: {', '.join(f'{d:.2f}' for d in exp_drops)})")
        print(f"{dom:12} expansion forced range {min(exp_forced):.2f}-{max(exp_forced):.2f}")
    if con_drops:
        b = boot_mean(con_drops)
        print(f"{dom:12} conservative harm {b[0]:.2f} [{b[1]:.2f},{b[2]:.2f}]")
print("\nWANDS reference: expansion 0.74 [0.64,0.83]; conservative 0.54 [0.46,0.61]")
