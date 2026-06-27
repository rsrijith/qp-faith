# Relevance validation — fill this in (~15-20 min)

Open `relevance_eval.csv` in any spreadsheet app. Each row has a **search query** and a **product title**. For each row, fill the `relevant_yes_no` column with **yes** or **no**:

> Would a shopper who typed this query reasonably want this product shown in the results?

Judge naturally, the way a shopper would. Don't look at product categories or overthink it. A "dinosaur" shopper who'd be happy to see a dinosaur rug, dinosaur bedding, or dinosaur wall art (not only dinosaur toys) should mark those **yes**.

Notes:
- ~60 rows. A few are obvious unrelated products (attention checks) — just answer honestly; they should mostly be "no".
- Leave a row blank only if you genuinely can't tell.
- Optional but stronger: have one other person fill a copy of the first ~25 rows too (save as `relevance_eval_annotator2.csv`); this gives an inter-annotator agreement number.

When done, save and tell me. I run `python scripts/score_human_eval.py` and it computes:
- the fraction of motif-matched products you judged relevant (the validation: confirms the probe's relevant set matches human relevance, so the measured harm is real),
- the attention-check pass rate,
- inter-annotator agreement if a second file exists,
and I write the result into the paper (Section 3.4 / Limitations).

Do NOT edit `relevance_key.csv` (the hidden answer key for attention checks).
