# Consolidation decision after final verification and matching feasibility

No manuscript edits, no new GPU training, no outcome-driven EOS weight/LR/model search.
This records work performed after the completed36-training EOS control.

## Final verification: PASS

`tmlr_eos_final_verification.json` checks all36 completed trainings and72 evaluations:
no discrepancies; all72 evaluation fingerprints recomputed;21 cached model/tokenizer
files hashed against the frozen inventory; raw primary scoring, generated-token counts,
costs, truncation, conservative loop rates and per-seed analysis values rederived.
The scheduling-only source amendment is accounted for. This is reproducible main-agent
verification, not independent human or publication-final scientific review.

No expensive training rerun is justified by this check. The earlier diagnostic label
`weighted_terminal_examples` remains dataset eligibility, not consumed examples; the
reported60-step dose is960 nominal examples. Preserve that distinction in paper methods.

## Length/token-budget control: redundant under the frozen screen

The original arms were already matched on Qwen trace-length decile x source x type.
Correct reconstruction confirms identical bin counts. Four repeated UUID rows require
full trace identity for this reconstruction; a UUID-only diagnostic was corrected,
without changing any scientific training/evaluation artifact.

Canonical masked target-length mean differences versus random:

- Qwen high -0.137%, low +0.109%; absolute standardized differences <0.007.
- Mistral high -0.507%, low +0.572%; absolute standardized differences <0.027.

Actual prepared Trainer sampler exposes a SeedableRandomSampler with initial_seed equal
to the recorded data_seed. Under that verified single-process sampler replay, first960
example exposure differences versus random range from -0.575% to +0.722% across both
families and all three seeds. Length totals agree with completed training records.
This is source/software-based deterministic replay, not a falsely claimed original
online per-batch logging stream. Means inside the frozen1%/0.1-SMD screening margins
do not prove exact per-step equality or causal comparability.

A new length-only bank would mostly repeat an existing control; do not run it by default.
Original reflection-density contrast remains sizeable (bank means approximately6.79,
2.59 and4.59 per1000 reference tokens for high, low and random).

## Strict same-question control: infeasible under the frozen design

Inspected all93,733 rows in the original cached source (no downloads/substitution).
Operator-preserving question matching finds11,495 source rows on archived pool
questions and2,436 question groups with multiple verified eligible source generations.
Only **nine** distinct questions meet every frozen pairing condition: two different
correct/reasoning-complete/stop-terminated traces, reflection-density gap>=2, canonical
supervised-length gap<=1% in both tokenizers, original length/context limits and external
decontamination. The minimum was960 distinct questions.

Do not relax margins after seeing this count, truncate/rewrite targets, repeat nine
problems through60 steps, or advertise question difficulty as controlled. Retain the
narrower composition-associated boundary claim. This is infeasibility of this fixed
archived-data design, not proof that every possible question-matched experiment is
impossible. The provisional word-key source report is retained; exact/operator-preserving
v2 is authoritative.

## Human validation: prepared, not completed

The blinded bundle contains127 items:95 loop assessments and32 final-option
interpretation cases. All labels remain empty. Instructions are in
`HUMAN_REVIEW_INSTRUCTIONS.md`; reviewer items/templates are in `data/human_review_v1`.
Explicit conditions, automatic labels and gold are withheld in reviewer items; textual
style can still reveal clues. A private key remains separate and must not be shared
before independent labeling. The loop sample is stratified, the choice sample purposeful;
neither unweighted aggregate is a new unbiased benchmark accuracy estimate.

Two real reviewers are preferred. One reviewer must be disclosed and cannot yield
inter-rater agreement. AI labels are not human validation. `analyze_human_review.py`
validates coverage/labels/identifiers and requires human-review attestation; it cannot
prove identity or manufacture agreement. Population estimates need a separate weighted
uncertainty analysis after actual labels arrive. Do not claim it has already happened.

## Next boundary

The separate bounded teacher-forced EOS diagnostic is now complete; see
`tmlr_eos_effectiveness_probe_summary.md`. The added scalar term is<0.17% in all42
sampled base/final arm/checkpoint conditions. This addresses part of the intervention-
strength interpretation gap, not the missing historical gradient logging or human review.
The verified nonweight raw evidence/archive is also now local; see
`tmlr_reproducibility_manifest_2026-10-09.json`.

Consolidate verified results and limitations; do not launch more experiments by default.
The useful current story is a replicated but partly prompt-sensitive composition-
associated cap/quality boundary, with a negative fixed EOS intervention—not a validated
attribution method, causal reflection effect, universal stable efficiency gain or
guaranteed TMLR acceptance. Human labels and author-led manuscript revision remain.
