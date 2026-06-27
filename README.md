# QP-Faith: Query-Planner Faithfulness benchmark and harm protocol

Public artifacts for the paper "Mis-Routed, Not Hallucinated: How LLM Query Planners Over-Constrain E-commerce Search, and QP-Faith, a Benchmark and Fix."

This repository accompanies the paper (under review). It contains the benchmark, probe generators, model outputs, scoring code, and figures. The manuscript and its arXiv preprint will be linked here on publication.

## What this measures
Whether an LLM query planner emits structured attribute filters the query never expressed, and whether applying them harms retrieval. Headline: for type-less entity queries planners route the term into the product category slot; as a hard pre-filter this removes 0.62-0.74 of relevant recall, across nine models / six providers (plus two further Google models on WANDS), with a schema-change fix.

## Contents
- `data/` — WANDS + ESCI preparation; `data/probe/` (procedural entity+attributed probe, labels known by construction); `data/probe_esci/` (second-domain probe); `data/gold/` (auto query-attribute reference + verification packet); `data/human_eval/` (relevance-validation sheet + key).
- `scripts/`
  - `make_probe.py`, `make_probe_esci.py` — procedural probe generators (WordNet motif mining).
  - `make_gold.py` — automatic query-attribute reference + verification packet.
  - `run_pilot.py` (local MLX planners), `run_frontier.py` (Anthropic/Gemini/OpenAI/Together/Groq/Mistral APIs), `run_probe.py`, `run_esci.py`.
  - `score_v2.py` (slot grading + cluster bootstrap), `score_probe.py`, `score_esci.py`, `score_recovery.py` (detector mitigation), `score_soft.py` (soft penalty).
  - `validate_retrieval.py`, `validate_dense.py`, `validate_enforcement.py` (real-retrieval harm, BM25 + dense), `validate_esci_probe.py`.
  - `baselines_detector.py` (non-LLM baseline, detector precision/recall, base rate, recall-fallback test), `score_human_eval.py` (relevance validation), `make_figs.py`, `seg_report.py`, `integrity_audit.py`, `assemble.py`.
- `results/` — per-run raw generations (`*__plans.jsonl`), scored rows (`*.parquet`), metrics (`*.json`), manifests, figures (`fig1_pipeline.png` pipeline, `fig2_universality.png`, `fig3_slotlocus.png`, `fig4_enforcement.png`, `fig5_ablation.png`, `fig6_dose_response.png`).

## Reproduce
```
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install mlx-lm pandas numpy pyarrow rank_bm25 nltk sentence-transformers matplotlib anthropic google-generativeai openai
# data
( cd data && for f in query product label; do curl -sSLO https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/$f.csv; done )
python scripts/prep_esci.py
python scripts/make_probe.py && python scripts/make_probe_esci.py && python scripts/make_gold.py
# local planners (one model at a time, 24GB)
PYTHONHASHSEED=0 python scripts/run_probe.py --model <mlx-4bit path> --tag probe-qwen4-exp --mode expansion
python scripts/score_probe.py --tag probe-qwen4-exp && python scripts/validate_retrieval.py --tag probe-qwen4-exp
# integrity (run inside .venv; needs pandas)
python scripts/integrity_audit.py
```
Determinism: greedy (temp=0), fixed seeds, pinned HF revisions; every run writes a manifest (model, quant, dataset hash, prompt hash, retriever, seed). All paper numbers regenerate from `results/`.

## License / data
Code: MIT (proposed). Data: WANDS and ESCI are released by their owners under their own terms; we redistribute only derived probe queries and scoring outputs, not the source catalogs.
