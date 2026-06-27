#!/usr/bin/env python
"""Build the FULL Amazon/ESCI catalog + max motif probe (all available US product
shards), writing to NEW paths so the running 348-probe pipeline is undisturbed.
Output: data/probe_esci/catalog_full.parquet, data/probe_esci/probe_full.jsonl"""
import os, re, json, collections
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from nltk.corpus import wordnet as wn
OUT=os.path.join(os.path.dirname(__file__),"..","data","probe_esci")
MOTIF_LEX={"noun.animal","noun.plant"}
BLOCK={"male","female","down","blade","blades","bulb","bulbs","body","head","hand","back","stickers","sticker","mount","charger","simple","queen","human","needle","crop","giant","acer","mouse","heather"}
def is_motif(w):
    if w in BLOCK: return False
    ss=wn.synsets(w,pos=wn.NOUN); return bool(ss) and ss[0].lexname() in MOTIF_LEX
STOP=set("a an the of for with and or in on to set sets by your our new large small pack count".split())
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def words(s): return [w for w in norm(s).split() if w and w not in STOP and len(w)>=4 and w.isalpha()]

catf=f"{OUT}/catalog_full.parquet"
if os.path.exists(catf):
    df=pd.read_parquet(catf); print(f"reuse full catalog: {len(df)}")
else:
    files=[f for f in list_repo_files("tasksource/esci",repo_type="dataset") if f.endswith(".parquet") and "/" in f]
    print("esci parquet shards:",len(files))
    parts=[]
    for f in files:
        try:
            p=hf_hub_download("tasksource/esci",f,repo_type="dataset")
            d=pd.read_parquet(p,columns=["product_id","product_title","product_locale"])
            d=d[d.product_locale=="us"][["product_id","product_title"]]
            parts.append(d); print(f"  +{f}: {len(d)}")
        except Exception as e: print(f"  skip {f}: {e}")
    df=pd.concat(parts).drop_duplicates("product_id").dropna().reset_index(drop=True)
    df["ntitle"]=df["product_title"].map(norm)
    df.to_parquet(catf); print(f"FULL ESCI us catalog: {len(df)} unique products")

tok=collections.Counter()
for t in df["ntitle"]:
    for w in set(words(t)): tok[w]+=1
motifs=[w for w in tok if tok[w]>=10 and is_motif(w)]
motifs=sorted(motifs,key=lambda w:-tok[w])  # no cap: all clean motifs
titles=df["ntitle"].tolist(); pids=df["product_id"].tolist()
rows=[]
for w in motifs:
    rel=[pids[i] for i in range(len(pids)) if re.search(rf"\b{re.escape(w)}\b",titles[i])]
    if len(rel)<8: continue
    rows.append(dict(query=w,family="A_entity",known_slots={},relevant_pids=rel,n_relevant=len(rel),n_distinct_class=-1))
with open(f"{OUT}/probe_full.jsonl","w") as fh:
    for r in rows: fh.write(json.dumps(r)+"\n")
print(f"FULL probe: {len(rows)} motifs, {sum(r['n_relevant'] for r in rows)} judgments")
json.dump(dict(catalog=len(df),n_motifs=len(rows)),open(f"{OUT}/manifest_full.json","w"),indent=1)
