#!/usr/bin/env python
"""Relocate robustness/control analyses from the main text into a new Appendix C
(verbatim, so every number is preserved) and leave one-line pointers in the main text,
to bring TMLR main-content under 12 pages. Also renumbers tables to the new appearance
order. Run once; idempotent-guarded by checking for the Appendix C marker."""
import re, sys, os
P = os.path.join(os.path.dirname(__file__), "..", "PAPER.md")
t = open(P).read()
if "## Appendix C." in t:
    print("Appendix C already present; aborting to avoid double-run."); sys.exit(0)

def cut(text, start, end, label):
    """extract the inclusive span [start..end]; return (new_text, span)."""
    i = text.find(start)
    assert i >= 0, f"START not found: {label}: {start[:50]}"
    j = text.find(end, i)
    assert j >= 0, f"END not found: {label}: {end[:50]}"
    j += len(end)
    return text[:i] + "\x01" + label + "\x01" + text[j:], text[i:j]

# (label, start, end, pointer-text that replaces the span in main)
SPANS = [
 ("S_taxsweep",
  "The magnitude depends on filter strictness, and survives a taxonomy-aware filter.",
  "not the literal plant or food.",
  "The magnitude depends on filter strictness and survives a taxonomy-aware filter (pooled loss 0.58 at a strict semantic threshold, 0.30 moderate, 0.08 loose) and is unchanged when the 23 plant/food motifs are removed (decorative 0.73 vs plant/food 0.77); details in Appendix C.3."),
 ("S_multiclass",
  "**Multi-class control.** For organic WANDS",
  "| Gemma-2-9B | 71 | 129 |",
  "On organic WANDS product-type harm, 60-80% of harmful cases are multi-class artifacts and 20-39% clean (Appendix C.2), which is why the by-construction probe, not organic examples, carries the headline."),
 ("S_taxonomy_errors",
  "### 5.4 What gets injected: an error taxonomy",
  "| Defensible (ambiguous motif) | bulb -> light bulb, feeder -> bird feeder | light bulb, bird feeder |",
  "### 5.4 What gets injected\nThe 493 category injections (nine models, WANDS expansion) fall into four patterns: literal-entity misread (the majority: dinosaur->toy, pineapple->food), depiction-specific class (bee->jewelry), generic catch-all category (owls->home decor), and a handful of defensible ambiguous motifs (bulb->light bulb, feeder->bird feeder). In every non-defensible pattern the motif-defined relevant set is largely excluded. The full taxonomy with per-type counts is in Appendix C.1."),
 ("S_nonllm",
  "**Non-LLM planner baseline.** To confirm",
  "| rule-based catalog-vocabulary extractor | 0.25 | 1.00 |",
  "A non-LLM rule-based control (a catalog-vocabulary extractor that emits a product_type only when a known category token literally appears in the query) emits a type for only 25% of type-less entity queries, versus ~100% for the LLM planners under expansion, and the correct type for 100% of attributed queries: the failure is LLM-specific, not a property of structured planning (Appendix C.4)."),
 ("S_detprofile",
  "**Detector profile.** The query-type detector itself",
  "substantial (not perfect) protection on entity queries.",
  ""),
 ("S_selfcons",
  "**Self-consistency does not flag the failure.**",
  "which is the second reason the schema fix is primary.",
  "A free abstention signal does not exist here: sampling five plans per query, cross-sample agreement does not separate type-less from concrete queries (separation AUC 0.47), so an uncertainty threshold cannot gate the filter and the working interventions are explicit (the schema change and detector). The detector's own profile (recall 1.00 on concrete queries, precision 0.56, out-of-vocabulary degradation) is in Appendix C.5-C.6."),
 ("S_fallback",
  "**A low-result fallback gives uneven, catalog-dependent protection.**",
  "not a general fix.",
  "A low-result (query-relaxation) fallback that drops the filter when its result set is too small gives uneven, catalog-dependent protection (recovering 32-79% of the loss on WANDS, but rarely firing on a large catalog where guessed categories stay well-populated); it is a partial, catalog-specific mitigation, not a general fix (Appendix C.7)."),
 ("S_prompthyg",
  "### 5.7 Prompt sensitivity and hygiene",
  "which saturates near 1.00.",
  "### 5.7 Prompt sensitivity\nThe broader spurious-constraint rate over all 480 organic WANDS queries (not just the entity probe) also rises with prompt aggressiveness, to 0.31-0.88 at expansion (Appendix C.8); Mistral's 3.5-4.8% JSON parse-failure rate is reported there so its lower slot counts are not read as fidelity."),
 ("S_nondecor",
  "**The routing is not specific to home decor.** On 25 type-less motifs",
  "the routing itself does not depend on the domain.",
  "The routing is also not specific to home decor: on 25 type-less motifs from non-decor domains (occasions, activities, abstract themes) all four local models emit a product_type 1.00 of the time under expansion (Appendix C.9). What is home-decor-favorable is the harm magnitude, not the routing."),
]

appendix_blocks = {}
for label, start, end, ptr in SPANS:
    t, span = cut(t, start, end, label)
    appendix_blocks[label] = span
    t = t.replace("\x01" + label + "\x01", ptr)

# assemble Appendix C in a sensible order with subsection headers
ORDER = [
 ("C.1 Error taxonomy of injected categories", "S_taxonomy_errors"),
 ("C.2 Multi-class control (organic WANDS)", "S_multiclass"),
 ("C.3 Filter strictness and the taxonomy-aware sweep", "S_taxsweep"),
 ("C.4 Non-LLM rule-based control", "S_nonllm"),
 ("C.5 Detector profile", "S_detprofile"),
 ("C.6 Self-consistency does not flag the failure", "S_selfcons"),
 ("C.7 Low-result fallback", "S_fallback"),
 ("C.8 Prompt sensitivity over organic queries", "S_prompthyg"),
 ("C.9 Routing is not specific to home decor", "S_nondecor"),
]
def strip_lead_header(s):
    # drop a leading "### 5.x ..." or "**Bold.**"/"**Bold ...**" lead-in so the appendix uses its own header
    s = re.sub(r"^###[^\n]*\n", "", s)
    s = re.sub(r"^\*\*[^*]+\*\*\s*", "", s)
    return s.strip()
appx = ["## Appendix C. Additional controls and robustness analyses",
        "These analyses are referenced from Sections 5.2-5.7 and support the main results; each is relocated here for length.\n"]
for title, label in ORDER:
    appx.append(f"### {title}")
    appx.append(strip_lead_header(appendix_blocks[label]))
    appx.append("")
appx_text = "\n".join(appx) + "\n"

# insert Appendix C right before ## References
t = t.replace("## References", appx_text + "## References", 1)

# --- renumber tables to new appearance order ---
# find caption order in the (relocated) text
caps = re.findall(r"^Table (\d)\.", t, flags=re.M)
order = []
for n in caps:
    if n not in order: order.append(n)
# build old->new map by appearance
newmap = {old: str(i+1) for i, old in enumerate(order)}
print("table renumber (old->new, by appearance):", newmap)
# swap every 'Table N' (prose + captions) via temp tokens to avoid collisions
for old in newmap:
    t = t.replace(f"Table {old}", f"Table \x02{old}\x02")
for old, new in newmap.items():
    t = t.replace(f"Table \x02{old}\x02", f"Table {new}")

open(P, "w").write(t)
print("done. relocated", len(SPANS), "spans into Appendix C.")
