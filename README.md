# QP-Faith: a label-free audit for category mis-routing in LLM query planners

Public artifacts for the paper **"Category Mis-Routing in LLM Query Planners: A Reproducible Failure Mode and a Label-Free Audit for E-commerce Search"** (under double-anonymous review). No preprint is posted while the paper is under review.

## What this measures

Whether an LLM query planner emits a structured attribute filter the query never expressed, and what that costs retrieval when the filter is enforced before ranking.

For a **type-less entity query** naming a motif ("dinosaur", "pineapple"), planners route the term into the `product_type` slot. Applied as a hard category pre-filter, that single guess removes most of the motif-relevant catalog.

Headline results:

| finding | value |
|---|---|
| routing into the category slot, aggressive prompt, 9 planners / 6 providers | ~1.00 |
| WANDS recall@100 loss, restrained prompt | 0.54 |
| WANDS recall@100 loss, aggressive prompt | 0.74 |
| ESCI full-catalog mis-routing loss (brand / colour) | 0.032 / 0.085 |
| the same, hierarchical CI over planners and queries | [0.008, 0.057] / [0.030, 0.140] |
| ESCI per-fire exclusion where a guess fires (brand / colour) | 0.306 / 0.353 |
| a standard groundedness grader (RAGAS) passes mis-routings as faithful | 0.57 |
| routing after a one-line omit instruction | 0.16-0.27 |

The audit is **label-free**: it needs catalog metadata and a lexical taxonomy, not relevance judgments.

## Result-to-script mapping

Every number in the paper is produced by one of these scripts (this table mirrors Appendix D of the manuscript).

| result | script |
|---|---|
| probe routing and the prompt ladder, incl. organic-WANDS and 1,162-motif replications | `run_conditions.py`, `score_conditions.py`, `organic_ladder.py` |
| probe adjudication (type-less purity) | `probe_typeless_prelabel.py`, `build_adjudication_sample.py`, `score_adjudication.py`, `clean_motif_subset.py` |
| ESCI exclusion, routing, coverage, sensitivity, full-catalog retrieval | `esci_routing.py`, `esci_facet_exclusion.py`, `facet_coverage.py`, `esci_sensitivity.py`, `esci_dearn.py`, `decoupled_relevance.py` |
| slot x enforcement matrix and hierarchical bootstrap | `slot_enforcement_matrix.py`, `hier_bootstrap.py` |
| mitigation, query-type classifier, dissociation, faithfulness graders | `mitigation_by_type.py`, `query_type_classifier.py`, `dissociation_fullprec.py`, `faithfulness_check.py`, `ragas_groundedness.py`, `implicit_entailment_set.py` |
| cross-domain harm and prevalence | `clean_category_harm.py`, `typeless_prevalence.py`, `framework_exposure.py` |
| text-to-SQL and LangChain probes | `micro_sql.py`, `micro_sql_multi.py`, `langchain_selfquery.py` |
| freshness-gate audit | `integrity_audit.py` |
| figures | `remake_figs_r1.py` |

## Contents

- `data/` — WANDS and ESCI preparation; `data/probe/` (procedural entity + attributed probe, labels known by construction); `data/probe_esci/` (second-domain probe); `data/gold/` (automatic query-attribute reference and verification packet); `data/human_eval/` (relevance-validation sheet and key).
- `scripts/` — probe generators, planner runners, scorers, retrieval validators, mitigation and grader experiments. See the mapping above.
- `results/` — per-run raw generations (`*__plans.jsonl`), scored metrics (`*.json`), and manifests.

## Reproduce

```
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install mlx-lm pandas numpy pyarrow rank_bm25 nltk sentence-transformers matplotlib anthropic openai ragas

# data
( cd data && for f in query product label; do curl -sSLO https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/$f.csv; done )
python scripts/prep_esci.py
python scripts/make_probe.py && python scripts/make_probe_esci.py && python scripts/make_gold.py

# routing census and the prompt ladder
python scripts/run_conditions.py --tag probe-qwen4-exp --mode expansion
python scripts/score_conditions.py

# full-catalog ESCI retrieval harm (cached BM25 index; CPU only)
python scripts/esci_dearn.py --version large --k 1000

# pooled intervals that resample planners as well as queries
python scripts/hier_bootstrap.py
```

Determinism: greedy decoding (temperature 0), seed 13, pinned model revisions, 4-bit local quantization. Every run writes a manifest recording the model revision, quantization, seed, prompt hash, dataset hash, and retriever.

Two senses of reproducibility are distinguished. The persisted plans and scores regenerate the reported numbers **deterministically**, and that is what this repository releases. Re-calling a hosted API is instead an **independent behavioral replication** subject to provider drift.

## Licence and data

Code: MIT. WANDS and ESCI are released by their owners under their own terms; only derived probe queries and scoring outputs are redistributed here, never the source catalogs.
