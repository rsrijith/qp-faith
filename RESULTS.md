# QP-Faith - Released Results Record

Auto-generated from persisted artifacts via `scripts/integrity_audit.py`. Must match the manuscript.

## Probe hashes (analyzed sets)
- WANDS probe (`data/probe/probe.jsonl`): 55 entity queries. sha256 `6d01c359119065ce03bf6248b4bc1a347c9c590eacc58ac04fdc1ba27e90fd83`
- Amazon/ESCI probe (`data/probe_esci/probe_full.jsonl`): 1162 entity queries. sha256 `1a92414cf346b1d1f9e9ed0a474d53ccfdbd643a5f56dd86667c2eae09587768`

## Headline (review round, 2026-06-27)
- Realized recall@100 loss is reported per PROMPT with a CLUSTER bootstrap over the 55 query clusters (not the 493 model-query pairs, which would pseudo-replicate): **conservative 0.54 [0.46, 0.61]**, **expansion 0.74 [0.64, 0.83]** (`headline_stats_byprompt.json`, `headline_stats_v2.py`). Per-model conservative loss 0.24-0.74 (`con_drops.json`).
- Taxonomy-aware filter is threshold-dependent (`taxonomy_tau_sweep.json`): pooled loss 0.58 (strict tau=0.7) / 0.30 (0.6) / 0.08 (loose 0.5). Not a clean fix.
- Routing is domain-general: 1.00 forced on 25 non-decor motifs, 4 local models (`nondecor_routing.json`).
- Self-consistency does NOT flag the failure (AUC 0.47, `selfconsistency_summary.json`).
- Schema-vs-prompt dissociation: optional slot leaves routing 1.00; omit-instruction drops to 0.16-0.27 (`ablation_dissociation.py`).

## Supersession notes
- n=31->55 WANDS and ESCI n=60->1162 corrected; freshness gate in integrity_audit.py.
- "Gemini Flash resists" was a superseded artifact; all models collapse under expansion.
- The earlier pooled CI [0.71,0.78] was PSEUDO-REPLICATED (493 treated as independent); corrected to a query-clustered [0.64,0.83] (+-~0.10).

## Current values
```
CURRENT HEADLINE VALUES (n-correct)
========================================================================

[3.1] WANDS catalog: products=42994; probe A_entity=55 B_attr=18; ESCI probe=1162

[Table 1] forced product_type rate (con / exp), n=55:
   Qwen2.5-7B           con 0.64   exp 1.00 [1.00,1.00]
   Llama-3.1-8B         con 0.98   exp 1.00 [1.00,1.00]
   Mistral-7B           con 0.98   exp 1.00 [1.00,1.00]
   Gemma-2-9B           con 0.33   exp 1.00 [1.00,1.00]
   Haiku 4.5            con 0.95   exp 1.00 [1.00,1.00]
   Sonnet 4.6           con 0.49   exp 0.98 [0.95,1.00]
   Llama-3.3-70B        con 0.91   exp 1.00 [1.00,1.00]
   gpt-4o               con 0.87   exp 0.98 [0.95,1.00]
   gpt-5.2              con 0.62   exp 1.00 [1.00,1.00]
   --> expansion band: 0.98-1.00

[Table 2] real-retrieval recall@100 drop (exp), n=55:
   Qwen2.5-7B           0.737 [0.628,0.836]  (n_typed=55)
   Llama-3.1-8B         0.754 [0.656,0.848]  (n_typed=55)
   Mistral-7B           0.759 [0.660,0.852]  (n_typed=55)
   Gemma-2-9B           0.723 [0.614,0.824]  (n_typed=55)
   Haiku 4.5            0.745 [0.637,0.843]  (n_typed=55)
   Sonnet 4.6           0.738 [0.630,0.840]  (n_typed=54)
   Llama-3.3-70B        0.733 [0.633,0.828]  (n_typed=55)
   gpt-4o               0.714 [0.602,0.816]  (n_typed=54)
   gpt-5.2              0.761 [0.650,0.859]  (n_typed=55)
   --> recall-loss band: 0.71-0.76

[App. A] enforcement hard-filter (lambda 1.0..0.0) recall@100 loss:
   Qwen bm25: 0.00 0.00 0.00 0.00 0.00 0.74   (n=55)
   Qwen dense: 0.00 0.01 0.08 0.11 0.11 0.74   (n=55)
   Llama bm25: 0.00 0.00 0.00 0.01 0.01 0.75   (n=55)
   Llama dense: 0.00 0.03 0.32 0.49 0.49 0.76   (n=55)

[5.5] schema-nullability ablation: exp(forced) -> opt(forced), n=55:
   Qwen: 1.00 -> 0.16
   Llama: 1.00 -> 0.18
   Mistral: 1.00 -> 0.24
   Gemma: 1.00 -> 0.27
   --> schema-fix band: 0.16-0.27

[5.5] cross-encoder rerank survival (no-filter -> hard-filter recall):
   qwen4: 0.811 -> 0.611 (loss 0.200, n=55)
   llama4: 0.811 -> 0.493 (loss 0.318, n=55)
   mistral4: 0.811 -> 0.381 (loss 0.430, n=55)
   gemma4: 0.811 -> 0.426 (loss 0.385, n=55)
   --> rerank-survival band: 0.20-0.43

[5.6] ESCI (Amazon 1.2M catalog), forced + real recall@100 drop, n=1162:
   Qwen2.5-7B           forced 0.98  recall_drop 0.44  (n=1162)
   Llama-3.1-8B         forced 1.00  recall_drop 0.49  (n=1162)
   Mistral-7B           forced 0.99  recall_drop 0.50  (n=1162)
   Gemma-2-9B           forced 0.99  recall_drop 0.47  (n=1162)
   Haiku 4.5            forced 1.00  recall_drop 0.39  (n=1162)
   Sonnet 4.6           forced 1.00  recall_drop 0.30  (n=1162)
   Llama-3.3-70B        forced 0.99  recall_drop 0.44  (n=1162)
   gpt-4o               forced 0.98  recall_drop 0.41  (n=1162)
   gpt-5.2              forced 1.00  recall_drop 0.22  (n=1162)
   --> ESCI forced 0.98-1.00; recall-drop 0.22-0.50

========================================================================
```
