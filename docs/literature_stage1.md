# Literature check for Stage 1 (search-based, 2026-10-02)

Method: eight web searches over arXiv, ACL Anthology, and proceedings sites. Claims below
come from search-result excerpts and abstracts, not full reads. Verify each paper before
citing it. Absence of a hit is weak evidence of a gap, not proof.

## Closest work

| Work | What it does | Relation to ILA |
|---|---|---|
| Infra, "Which Data Attributes Stimulate Math and Code Reasoning?" (NeurIPS 2025, arXiv 2505.19949) | Influence functions on long-CoT SFT data (Bespoke-Stratos-17k, Qwen2.5-7B-Instruct); instance, sequence, and token level; target is reasoning accuracy | Same machinery, different target (accuracy, not generated tokens). Finds that exploration after a correct answer helps accuracy, so cutting reflection is not free. |
| Segment-Level Attribution for Selective Learning of Long Reasoning Traces (arXiv 2602.00425, ICLR 2026) | Integrated-gradient attribution to the final answer; masks the loss on unimportant segments; 5 to 18% shorter outputs | Attribution is inside a trace and toward answer probability, not toward deployment cost across a dataset. A direct efficiency baseline for any pruning demo. |
| SELECT2REASON (ACL Findings 2026) | Data selection on OpenR1-Math-220k from question difficulty plus a trace-length heuristic; 10% of data matches full data | Same pool as Stage 1. Its length heuristic is a baseline the Stage 1 features already cover. |
| Don't Overthink It (arXiv 2505.17813) | Training on the shortest of several traces: 5.8% fewer thinking tokens, +2.8% accuracy over random selection | Simple length-based selection is a published baseline for the pruning demo. |
| Datamodels (Ilyas et al., 2022), DsDm (Engstrom et al., 2024) | Linear regression from training-subset masks to model outputs; dataset selection from it | The natural attribution baseline for Stage 3. Neither applies it to generation length, as far as the search shows. |
| Form Over Content in Gradient-Based Data Attribution (arXiv 2609.19589) | TracIn-style gradient scores track format and length more than content | Direct risk for TraCIL: its scores may collapse to a length or format proxy. Needs an explicit gradient-score-versus-length comparison. |
| When Internalization Fails (ACL Findings 2026) | Shortening reasoning makes models compensate with longer post-think text | Measure total generated tokens, as Stage 1 does, never only the visible reasoning span. |
| NoWait (arXiv 2506.08343), graph-based CoT pruning (ACL Findings 2026), TLDR, LS-Mixture SFT | Reflection-token suppression, pruning of redundant reflection, long/short mixing | Reflection density is a known cost lever, which makes it the strongest cheap heuristic. |
| Do Influence Functions Work on LLMs? (arXiv 2409.19998) | Influence functions perform poorly on LLMs; representation similarity does better | Supports a retraining-based ground truth and a RepSim-style baseline. |

## What the search did not find

No paper that defines inference cost (generated tokens) as the attribution target for
training data and validates it against group-removal retraining. This is the gap the
project would fill, subject to a full read of the papers above and a check of recent
preprints.

## Consequences for the plan

1. Report accuracy and tokens per correct answer with every cost result (Infra shows that
   reflective behaviour carries accuracy).
2. Treat length and reflection density as the main baselines, and add a gradient-score
   versus length comparison before claiming anything about TraCIL.
3. Any pruning demo must compare against shortest-trace selection and segment-level
   selective SFT.
