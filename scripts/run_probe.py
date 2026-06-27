#!/usr/bin/env python
"""Run a planner over the procedural probe (data/probe/probe.jsonl).
Reuses load_planner + PROMPTS from run_pilot. GPU job — run only when free.
Writes results/<tag>__probe_plans.jsonl."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from run_pilot import load_planner, PROMPTS

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=list(PROMPTS), default="expansion")
    ap.add_argument("--probe", default=f"{DATA}/probe/probe.jsonl")
    args = ap.parse_args()
    probe = [json.loads(l) for l in open(args.probe)]
    planner = load_planner(args.model, PROMPTS[args.mode])
    out = open(f"{RES}/{args.tag}__probe_plans.jsonl", "w", buffering=1)
    from run_pilot import extract_json
    t0 = time.time()
    for i, r in enumerate(probe):
        raw = planner(r["query"])
        out.write(json.dumps({**r, "mode": args.mode, "raw": raw,
                              "plan": extract_json(raw)}) + "\n")
        if (i+1) % 25 == 0: print(f"  {args.tag}: {i+1}/{len(probe)} ({time.time()-t0:.0f}s)")
    out.close()
    print(f"done {args.tag} ({args.mode})")

if __name__ == "__main__":
    main()
