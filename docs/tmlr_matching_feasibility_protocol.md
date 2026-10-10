# Matching feasibility protocol (2026-10-09)

Scope: CPU-only design assessment before any new training; no manuscript changes.
The original arms already match Qwen trace-length decile x source x problem type.
Their bank mean trace-length gap high/low is about0.25%; do not present this as an
uncontrolled gross-length comparison. Remaining possibilities are supervised-token
exposure in the actual sampled dose, within-bin imbalance, and question identity.

H1: original banks have a meaningful residual token/length imbalance. Operational
screen: relative mean differences >1% or absolute standardized mean differences >0.1
for measured supervised lengths; report distributions and all seeds, not a significance
search. Values inside the margins make a duplicate length-only control lower priority,
not proof of causal comparability or mathematical equivalence.

First assess all bank rows, normalized exact-question duplicates, UID overlap, lengths
and reflection densities, and source/type/decile counts. Then measure canonical masked
supervised lengths for each model using pinned tokenizers and the original serialization.
If actual training indices can be recovered/replayed reproducibly, separately assess
the first960-example dose. Do not label a guessed NumPy permutation as the actual
Trainer sampler, and do not substitute full-bank totals for consumed-token exposure.
No trace truncation, target rewriting, outcome-dependent selection, LR tuning, or new
evaluation results enter selection.

Same-question pilot is feasible only if at least960 distinct clean question groups
provide at least two different already-verified, reasoning-complete, stop-terminated
natural traces with reflection-density difference >=2 markers per1000 reference tokens,
canonical supervised-length difference <=1% in both model tokenizers, and original
context limits without truncation. These margins are frozen before inspecting cached
alternate generations; no relaxation follows an insufficient count.
Assess archived pool first; if that fails, inspect cached original multi-generation
source only, without downloading/substituting a new dataset based on model outcomes.
Retain the original train/eval exclusions and additionally filter against both frozen
external benchmarks before any new bank. One trace per existing UUID is not evidence
that alternate verified source generations do not exist. Report what was inspected.

If feasible, freeze a separate experiment protocol defining matched questions, target-
length and per-step token margins, seeds0/1/2, fixed recipes, original baseline/prompt
scoring, uncertainty and failure conditions before implementation or GPU launch. Random
control may mix high/low within each question; it is not automatically a third distinct
verified trace. Question matching still does not isolate reflection alone: trace content
and reasoning strategies can differ. If infeasible, save the counts and retain a narrow
composition-associated boundary claim. Do not manufacture a causal comparison.

Archive data/code hashes, feasibility metrics, rejected/retained counts and the decision.
Full model outcomes remain exploratory; no additional coefficient/model searches follow
an unfavorable feasibility assessment. Human labels require actual human review.

Implementation clarification before any new GPU comparison: same-question pair keys
fold whitespace only and preserve operators, variable case and LaTeX. Word-tokenized
normalization is useful for conservative overlap checks but must not establish exact
mathematical question identity. The provisional cached-source report using word keys
is retained; the operator-preserving v2 report is authoritative. Margins/minimum sample
size are unchanged. The first bank-bin reconstruction also needed full trace identity,
not UUID alone, because the archived pool has four repeated UUID rows; the corrected
bank report confirms identical original bin counts. Neither diagnostic issue changed
historical training or evaluation results.
