# Current claim-to-evidence map for author-led revision

No manuscript has been changed. Read `experiment_updates.json` and the latest registry
before revising prose. Exact numeric/seed values remain in the source JSON artifacts.

- **SUPPORTED WITH CAVEAT:** strong adaptation can create large composition-associated
  differences in capped generated-token cost while quality/termination degrade.
  Evidence: `tmlr_external_{qwen,mistral}_analysis.json`, and fresh EOS weight1 analyses.
  Scope: these model/recipe/math benchmark conditions, not universal efficiency effects.
- **SUPPORTED WITH CAVEAT:** large observed capped-cost gaps are predominantly associated
  descriptively with cap-hit frequency in the relevant degraded conditions.
  Evidence: signed `frequency_component` and `finished_length_component` keys in the
  external/EOS analysis contrasts. Algebraic decomposition, not causal mediation.
- **UNSUPPORTED:** all composition-cost gaps are prompt-robust.
  Evidence: `tmlr_format_control_comparison.json`; Qwen AQuA high-random LR1e-5/60
  changes +743.59 to -12.84 capped tokens, control CI spans zero. No formal interaction
  or equivalence test is supplied. Do not hide baseline Qwen accuracy drop under prompt.
- **UNSUPPORTED:** the fixed terminal-EOS weight10 control fixes the degradation.
  Evidence: `tmlr_eos_control_comparison.json`, `families/*/primary_gate`; both fail the
  frozen10pp cap-reduction criterion. Intervals span zero. Does not refute all possible
  termination mechanisms or prove equivalence; historical training gradients unlogged.
- **SUPPORTED WITH CAVEAT:** the added scalar terminal-EOS term is small on the fixed
  teacher-forced base/final probe sample. Evidence: `tmlr_eos_effectiveness_probe_*.json`;
  all42 condition shares<0.17%. Posthoc,32 train cases per arm; not a gradient-norm or
  free-running termination test, and not proof of the reason the primary control is null.
- **SUPPORTED WITH CAVEAT:** gross length and supervised-token exposure are already
  balanced under the frozen feasibility screen.
  Evidence: `tmlr_matching_pool_feasibility_v2.json` and
  `tmlr_supervised_token_exposure.json`. Original bin counts match; full-bank target means
  and fixed-sampler960-example dose means differ by <1% versus random. This is not proof
  of identical per-step gradients, exact question difficulty or causal reflection.
- **SUPPORTED WITH CAVEAT:** the fixed strict same-question design is infeasible from
  the inspected cached source under its declared criteria.
  Evidence: `tmlr_matching_cached_pairs_feasibility.json`; nine qualifying question pairs
  versus960 required, after93,733 source rows inspected. This is a feasibility result,
  not a model experiment or a claim that every alternative design is impossible.
- **UNVERIFIED:** reflection itself independently causes the cost changes.
  Exact-question/strategy confounds remain. No valid new matched GPU experiment was
  launched; the failed feasibility assessment cannot be described as causal evidence.
- **UNVERIFIED:** the repetition heuristic and secondary answer mapping are human-validated.
  `data/human_review_v1/manifest.json` is PREPARED_NOT_HUMAN_REVIEWED;127 reviewer items
  exist but templates are empty. A prepared sample is not annotations or agreement.
- **SUPPORTED WITH CAVEAT:** the completed EOS matrix passes final artifact/metric checks.
  `tmlr_eos_final_verification.json`:36 trainings,72 evaluations/fingerprints,21 model/
  tokenizer files, independently rederived raw metrics and seed summaries, no issues.
  Main-agent reproducibility verification is not independent publication-final review.
- **UNSUPPORTED/UNEVALUATED AS APPLICABLE:** a validated attribution estimator, general
  useful efficiency gain, pristine pretraining decontamination or guaranteed acceptance.
  Do not manufacture these claims from gate experiments or negative controls.

Keep original strict scores separate from gold-blind content diagnostics and new prompt
generations. Mark exploratory/posthoc/reused-benchmark evidence. Do not replace missing
human annotations, p-values, interaction tests, equivalence analyses or new experiments
with prose. Human-led manuscript revision and venue-policy review remain separate.
