# Prompt for two evidence-grounded LaTeX research drafts

Upload `paper_evidence_packet.json` with this prompt. If available, upload the repository
as well, excluding virtual environments, caches, and model weights. Do not send a prompt
without its evidence packet and expect ChatGPT to know the experimental results.

---

You are helping a human researcher prepare two research manuscripts from the attached
evidence packet and repository:

1. A full empirical manuscript for Transactions on Machine Learning Research (TMLR).
2. A shorter companion draft for a suitable, explicitly non-archival ICLR 2027 workshop
   on negative results, LLM reasoning, or evaluation. A specific accepted workshop has
   not yet been selected. Do not invent its name, acceptance, deadlines, or policies.

Produce complete LaTeX source files, not just an outline, abstract, or writing advice.
These are research drafts requiring human scientific review; never describe them as
submission-ready solely because you generated or compiled them.

## Publication policy and human authorship

Read the current official TMLR editorial/author policies and ICLR workshop policies.
TMLR permits reuse from venues explicitly declared non-archival in writing, but prohibits
overlapping results from archival peer-reviewed publications, including expanded
conference papers. Check the chosen workshop's own policy before recommending a sequence
or simultaneous submissions. Do not present the two drafts as two archival publications.

ICLR 2027 short/tiny-paper guidance requires primarily human-authored submissions and
permits AI assistance rather than AI-generated papers. Your workshop output is therefore
a working research draft for human verification and substantial author-led rewriting,
not a compliant final short/tiny submission. If a chosen track's policy prohibits the
requested drafting use, explain that limitation and provide permitted assistance instead
of claiming compliance. Do not invent exemptions. Include a factual proposed AI-assistance
disclosure for author review. Human authors must understand and take responsibility for
all claims, methods, citations, code, and results. Use anonymous authors where appropriate;
never invent author names, affiliations, funding, approvals, or acknowledgments.

## Scientific framing

### Study extension now in progress

The packet now includes completed post-hoc cost decompositions, the missing natural-data
builder/feature code and build report, an outcome-blind frozen SVAMP/AQuA manifest, and
the pinned Mistral serialization precheck. The external evaluations and second-family
replication are queued; those planning/precheck artifacts are NOT completed experimental
results. Do not invent their outcomes. Use new results only when their verified analysis
artifacts are actually supplied, and disclose any branch still incomplete.

Use the exact signed cost decomposition and confidence intervals from
`results/tmlr_existing_analysis/analysis.json`; do not describe it as causal mediation or
silently present finished-only cohorts as a controlled efficiency comparison. Existing
large gaps are dominated descriptively by cap-hit frequency. The new banks contain 400
SVAMP and 254 AQuA examples, not 500 each. Report benchmarks separately; any equal-benchmark
aggregate must use explicit weights and appropriate stratified uncertainty. AQuA primary
accuracy is explicit option-letter accuracy, with any option-value rescoring labeled
secondary. Compare token costs within model families: different tokenizers do not make
raw token totals physically equivalent across Qwen and Mistral. The fixed Mistral resource
settings differ from historical Qwen microbatch/context limits; model-family replication
is not an isolated causal architecture experiment.

The original project proposed inference-cost attribution to training data. An attribution
estimator was not evaluated in these gates. Write an empirical study of the conditions
under which apparent training-trace composition effects on output length arise and
whether they coexist with usable accuracy and termination.

Candidate framing, which you must qualify against the evidence:

“Natural training-trace composition produced substantial differences in capped generation
cost in a degraded regime; under the stable recipes tested, composition effects remained
statistically inconclusive.”

The contribution should be a carefully controlled boundary/failure analysis, not a claim
that composition never matters, not proof of equivalence, not a validated attribution
method, and not an efficient model obtained merely by reducing capped token counts.
Keep the scope to the tested models, datasets, adapter architecture, and recipes.

## Evidence hierarchy and numerical rules

- Read every numeric artifact and the relevant methods/code in the packet before drafting.
- Prefer raw predictions when supplied, then JSON artifacts, then generated Markdown.
  Trace every table cell, figure, and quantitative sentence to an artifact path and key.
- The packet has its own hashes; verify them if tools permit. Flag missing files and
  unresolved conflicts. Do not fill gaps with plausible values.
- Preserve units: accuracy/truncation values in JSON are fractions; differences expressed
  as percentage points are not relative percent improvements.
- Keep benchmark-specific and equally pooled outcomes distinct. The natural experiments
  evaluate 500 GSM8K and 500 MATH problems per run. Do not treat training seeds as additional
  independently sampled benchmark questions.
- Report point estimates, uncertainty, all seeds, accuracy, and truncation together.
  Do not count reused checkpoints as newly trained independent replications.
- A missing token count or missing confidence interval must be identified as missing,
  not fabricated. If you recompute anything, document the exact formula and input paths.
- Describe the actual bootstrap implementation, including prompt and seed resampling;
  do not call it a paired-seed bootstrap if seeds were sampled independently across arms.
- Evaluate the assumptions and limitations of the pooled-noise F test. Do not equate
  its non-significance with practical equivalence or absence of effect.
- No new experiments may be invented. Proposed experiments belong in future work and
  must be explicitly labeled unperformed.

## Mandatory numerical anchors (verify against the packet before using)

A. Canonical Qwen2.5-7B-Instruct base: equally pooled mean tokens 491.608, accuracy
81.60%, truncation 0.50%. Benchmark accuracies are GSM8K 91.0% and MATH-500 72.2%.

B. LR 1e-5, step 15, planned cosine horizon 230: 29 evaluated runs, 27 new trainings,
two reused random adapters. Arm B: between-subset variance 57.01314785, pooled seed-noise
variance 97.76892687 (10 df), F=0.5831417985, one-sided p=0.8328178705, relative cost
range 6.024385%. The range passes 5%; the significance criterion fails. The point
explainable fraction is clipped to zero; its prompt-bootstrap CI reaches 0.4295.
The stored zero heuristic share is not interpretable when explainable variance is zero.

Step-15 Arm A means: high 496.784333, low 501.295667, random 509.422 tokens.
High−random: −12.637667 tokens (−2.4808%), CI [−32.039208, +6.985775].
Low−random: −8.126333 tokens (−1.5952%), CI [−28.991258, +10.568342].
All other contrasts and all per-run values must come from the attached JSON.

C. LR 1e-5, step 60, same planned horizon: nine evaluated runs, seven new trainings,
two reused step-60 random adapters. Equally pooled arm means:
high: 4210.161333 tokens, 35.1667% accuracy, 62.90% truncation, 62.20% likely loops;
random: 3553.207667 tokens, 45.9667% accuracy, 49.7667% truncation, 48.9333% likely loops;
low: 2802.179333 tokens, 57.80% accuracy, 35.30% truncation, 34.40% likely loops.
High−random: +656.953667 tokens (+18.4890%), CI [+388.100367, +901.798058].
Low−random: −751.028333 tokens (−21.1366%), CI [−1012.468092, −516.094433].
Both contrasts pass the signal rule and relative-accuracy guard; truncation causes the
registered continuation stop. Calling this outcome “null” was a reporting error.

D. Final bounded attempt: LR 3e-6, step 60, planned horizon 230; nine new trainings.
All three engineering pilots passed. Two of nine confirmation run-gates failed, hence
two of 12 total recorded run-gates failed. Distinguish run-gates from per-benchmark checks.
High−random: +1.419333 tokens (+0.2674%), CI [−24.905633, +27.030033].
Low−random: +20.406667 tokens (+3.8445%), CI [−5.528450, +48.015783].
These rounded anchors must be checked against exact values in final_report.json.
Both intervals include zero. Pooled per-run accuracies range 80.9–82.2%, with pooled
truncation 0.2–1.0%; do not characterize this as another severe model collapse.
MATH-confirm400, high seed 2: accuracy .7125, truncation .0175, likely loops .0225.
MATH-confirm400, low seed 2: accuracy .705, truncation .025, likely loops .0325.
The loop ceiling is .02; truncation must be strictly below .02. The canonical MATH
confirmation accuracy is .7225. The final attempt failed strict quality gates and did
not demonstrate a reliable stable cost signal.

## Mandatory caveats

- Early 1.5B/plain-prompt failures and the later canonical-chat measurement repair must
  not be merged into one supposedly controlled intervention.
- Natural-composition models use LoRA rank 64, alpha 128, dropout .05, seven projection
  families; batch size 2 × accumulation 8 = effective batch 16. Step 15 exposes roughly
  240 examples, step 60 roughly 960; it does not train on the entire assigned subset.
- Arm A subsets contain 3,676 examples; Arm B contains 4,000. Length/source/problem-type
  matching does not eliminate every correlated trace or problem characteristic. Avoid
  claiming that the reflection-density feature itself is uniquely causal.
- Fourteen of sixteen Arm B subsets have only one seed. Noise estimation is limited.
- LR 3e-6/step 60 versus LR 1e-5/step 15 is not a pure dose comparison.
- Benchmarks, including the 400-example partitions, were previously observed. They are
  not pristine independent confirmation sets. State this explicitly, including tuning.
- Truncated generations are censored at 6144 tokens. Means are lower bounds on unrestricted
  length; differences under severe censoring cannot establish usable efficiency.
- Accuracy relies on boxed-answer extraction and the implemented approximate matcher;
  it is not a full symbolic equivalence evaluator. First-box rescoring did not rescue
  the degraded-regime accuracy. Conservative loop flags are diagnostics, not perfect labels.
- Audit fixes found an unused HF-adapter-loading bug and weak cache checks. The audited
  vLLM results remain valid evidence; do not claim they were all rerun after the fixes.
- New baseline checks in analysis_v2 are retrospective for old runs. Distinguish original
  stopping rules from later diagnostic analyses and prospective final-attempt gates.
- Protocols were documented prospectively in the repository; do not claim public registry
  preregistration without evidence of an actual registration.
- The exact training mechanism producing loops remains unresolved. A bounded single-example
  HF/PEFT check also hit the cap; this rules out a vLLM-only explanation for that example,
  not full backend equivalence. Model weights intentionally remain remote and are not in
  this upload. Do not claim publicly released weights or complete replication from this packet.

Synthetic Gate 0 experiments are optional context, not substitutes for natural-data evidence.
Use their four analysis JSONs to verify models, seed counts, template/paraphrase differences,
length matching, accuracy, and uncertainty. Do not blindly repeat early narrative conclusions
about wording or structure. The 0.5B analyses have six seeds and the 1.5B analyses have three;
verify all details before combining them. If they distract from the core paper, move them
to the appendix or omit them with an explanation.

## Literature and citations

Use browsing to verify primary papers, datasets, model documentation, and venue templates.
The repository's literature notes are search-based leads, not a verified bibliography.
Read the actual sources; never fabricate titles, authors, publication venues, DOIs, or
BibTeX. Verify that each citation supports its associated statement. Identify the closest
work on long-reasoning fine-tuning, training-data selection, inference efficiency,
overthinking/looping, and data attribution. Explain the difference without claiming an
unverified “first-ever” contribution. If a needed citation cannot be verified, omit that
claim or disclose the unresolved reference in a separate checklist.

## Deliverables and manuscript structure

Write two distinct but consistent drafts:

TMLR: title, abstract, introduction, focused related work, hypotheses/protocol timeline,
experimental methods, three-regime results, audit/alternative explanations, discussion,
limitations, reproducibility/data availability, conclusions, verified references, and
appendices containing per-seed metrics and complete gate details.

Workshop: one focused empirical lesson, concise methods, essential results, uncertainty,
limitations, verified references, and an appendix if the selected call permits it. Do not
guess a page limit or silently treat this as an ICLR main-conference manuscript.

Tables must display cost, accuracy, truncation, and uncertainty without hiding failed
conditions. Include the three regimes in a comparable table. Make plots from the actual
values; label axes and units and show seed variability or confidence intervals. Avoid
decorative figures. Figure captions must explain censoring and the relevant controls.

Return downloadable files named:

- `tmlr_paper.tex`
- `iclr_workshop_draft.tex`
- `references.bib` (verified references only)
- `claim_evidence_map.csv` (claim/table/figure, source path, JSON key/formula, caveat)
- `submission_readiness.md` (policy constraints, missing evidence, author-review checklist)

Each .tex file must be complete from document class through end of document. Use official
current templates only if actually retrieved. Include required style files and figure
assets so the build is self-contained. If templates cannot be retrieved, use a compilable
article-class review draft and clearly label it as not yet venue-formatted.

Where feasible, embed verified bibliography and numeric plots directly in each .tex using
filecontents and PGFPlots, so the sources remain usable without external image files.
Otherwise include every referenced asset and a reproducible figure-generation script.
Do not include unprovided files or dangling citations, labels, or cross-references.

Compile both papers with LaTeX if tools permit and inspect the outputs. Fix fatal errors,
missing references, misleading rounding, overflow, and unreadable figures. Supply PDFs
and compilation logs if produced. If a compiler is unavailable, explicitly say compilation
was not verified; never invent a successful build or a download link.

No TODOs or made-up facts in manuscript prose. Missing author identity can use an anonymous
review placeholder. Any substantive unresolved evidence must be disclosed rather than
written as established fact. Finish both drafts; do not stop after an outline. If tool or
context limits prevent completion, clearly list what remains rather than claiming delivery.

The final response should link the actual generated .tex files, any PDFs, bibliography,
and evidence map, and identify remaining scientific and submission-policy limitations.
If downloads cannot be created, return the full LaTeX contents in separately named code
blocks and state that actual files were not generated.
