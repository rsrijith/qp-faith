#!/usr/bin/env python
"""Stream McAuley Amazon-Reviews-2023 category fields for a domain and cache product_id -> category.

build_domain_probe.py kept only product_id + title (which is why the paper scores those domains with
a weak title instrument). Here we re-stream the same metadata and retain the real `categories`
hierarchy + `main_category`, restricted to the product_ids already in the domain catalog, so we can
measure a CLEAN-category harm (like WANDS's product_class) instead of the title proxy.

Writes data/probe_domain/<dom>/categories.parquet (product_id, category_path, main_category).
Restartable: skips if the parquet already covers the catalog. $0 (bandwidth only).

Run: ./.venv/bin/python scripts/domain_categories.py --category meta_Electronics --dom electronics
     ./.venv/bin/python scripts/domain_categories.py --category meta_Clothing_Shoes_and_Jewelry --dom apparel
"""
import os, json, argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "data", "probe_domain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--dom", required=True)
    args = ap.parse_args()
    ddir = os.path.join(ROOT, args.dom)
    outp = os.path.join(ddir, "categories.parquet")
    cat = pd.read_parquet(os.path.join(ddir, "catalog.parquet"))
    need = set(cat.product_id.astype(str))
    # also make sure we cover every relevant_pid referenced by the probe
    for l in open(os.path.join(ddir, "probe.jsonl")):
        r = json.loads(l)
        rp = r.get("relevant_pids")
        if isinstance(rp, str):
            try: rp = json.loads(rp)
            except Exception: rp = []
        need |= {str(p) for p in (rp or [])}
    print(f"{args.dom}: need categories for {len(need):,} product_ids", flush=True)

    if os.path.exists(outp):
        have = pd.read_parquet(outp)
        if need.issubset(set(have.product_id.astype(str))):
            print(f"  cache already covers all needed ids ({len(have):,}); done"); return

    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    path = f"datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/{args.category}.jsonl"
    ids, paths, mains = [], [], []
    got = set()
    n = 0
    with fs.open(path, "r") as fh:
        for line in fh:
            n += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            pid = str(r.get("parent_asin") or "")
            if pid in need and pid not in got:
                cats = r.get("categories") or []
                ids.append(pid)
                paths.append(" > ".join(str(c) for c in cats) if cats else "")
                mains.append(str(r.get("main_category") or ""))
                got.add(pid)
            if n % 200000 == 0:
                print(f"  streamed {n:,} rows; matched {len(got):,}/{len(need):,}", flush=True)
            if got >= need:
                break
    df = pd.DataFrame({"product_id": ids, "category_path": paths, "main_category": mains})
    df.to_parquet(outp)
    print(f"{args.dom}: streamed {n:,} rows -> categories for {len(df):,}/{len(need):,} product_ids -> {outp}")


if __name__ == "__main__":
    main()
