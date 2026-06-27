#!/usr/bin/env python
"""Run a planner over the ESCI slice (data/esci/queries.jsonl). Imports planner
from run_pilot (no main exec). Writes results/esci-<tag>__plans.jsonl. GPU job."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from run_pilot import load_planner, PROMPTS, extract_json
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=list(PROMPTS), default="expansion")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(f"{DATA}/esci/queries.jsonl")]
    planner = load_planner(a.model, PROMPTS[a.mode])
    out = open(f"{RES}/esci-{a.tag}__plans.jsonl", "w", buffering=1)
    t0 = time.time()
    for i, r in enumerate(qs):
        raw = planner(r["query"])
        out.write(json.dumps({"query_id": r["query_id"], "query": r["query"],
                              "mode": a.mode, "raw": raw, "plan": extract_json(raw)}) + "\n")
        if (i+1) % 50 == 0: print(f"  esci-{a.tag}: {i+1}/{len(qs)} ({time.time()-t0:.0f}s)")
    out.close(); print(f"done esci-{a.tag} ({a.mode})")
if __name__ == "__main__": main()
