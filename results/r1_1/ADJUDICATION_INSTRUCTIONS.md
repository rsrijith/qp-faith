# R1.1 motif adjudication — fill in the `human_label` column (~30-45 min)

Open `ADJUDICATION_SAMPLE.csv` in any spreadsheet. Each row is one search term from the probe.
For each row, put ONE of these in the **human_label** column (leave `notes` for anything unclear):

- **type-less** — the term names a decorative motif / theme / animal / plant / food / concept, NOT
  a product category. A shopper wants items *featuring* it, spanning many product categories.
  (dinosaur, rose, butterfly, flamingo)
- **genuine-product-type** — the term IS an actual product category that maps to one shelf.
  (sofa, lamp, doormat, backpack)
- **ambiguous** — could reasonably be read either way, or is context-dependent.
  (queen = bed size or royalty; simple = adjective; mint = color/plant/candy)

Judge as a typical general-catalog shopper would. The `llm_label` and `llm_rationale` columns show
the model's pre-label — you are ADJUDICATING it, so feel free to disagree; disagreements are the
point. Don't overthink; go with the natural shopper reading.

Rows: 150 total (55 WANDS, 95 ESCI).
The `random_flag` column marks rows drawn at random (for the agreement/purity estimate) vs targeted
candidate impurities — you fill in all of them the same way; the script handles the rest.

When done, save the file and tell me. I run `scripts/score_adjudication.py`, which computes:
- LLM-vs-human agreement (Cohen's kappa) on the random rows,
- the confirmed type-less purity of the probe with a bootstrap CI,
- the verified-clean subset, and re-reports the headline routing/harm on it (R1.1).
