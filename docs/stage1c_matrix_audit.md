# Stage 1c Natural-Composition Matrix: Independent Audit

Audit date: 2026-10-06

## Executive conclusion

The artifact and metric audit passes with no detected integrity failures. The registered decision is `STOP_NO_EFFECT`: under the tested 15-step, low-learning-rate recipe, natural subset composition does not produce a detectable output-cost effect beyond seed noise.

This is a low-dose null result, not evidence that composition cannot matter after longer training. Each newly trained adapter consumed 240 examples, only about 6.0–6.5% of its assigned 3,676–4,000-example subset.

## Integrity and provenance

- All 29 expected matrix run records are present and completed.
- The matrix contains 27 new training runs and two reused, validated step-15 adapters.
- All 29,000 raw prediction rows independently reproduce every reported count, accuracy, token statistic, and truncation rate.
- Evaluation UID order is identical across every run and the unadapted base evaluation for both GSM8K and MATH-500.
- Every evaluation used Qwen2.5-7B-Instruct, its recorded dynamic adapter, canonical-chat prompts, vLLM, greedy decoding, 500 examples per benchmark, and a 6,144-token generation cap.
- New runs reached exactly 15 optimization steps, retained a planned 230-step schedule, used learning rate `1e-5`, showed the expected scheduler state at step 10, contained no truncated training examples, and did not save merged models.
- The two reused adapters point to recorded step-15 trajectory checkpoints whose source runs completed 230 steps with canonical-chat prompting and learning rate `1e-5`.

## Independent statistical recomputation

Arm B recomputation exactly matches the analysis artifact:

- Between-subset variance: 57.0131
- Pooled seed-noise variance: 97.7689 with 10 degrees of freedom
- F statistic: 0.5831
- One-sided p-value: 0.8328
- Relative seed-0 subset cost range: 6.024%

The observed between-subset variance is smaller than estimated seed noise. The range clears 5%, but the significance condition fails; therefore the registered stop decision is appropriate.

The point estimate of explainable variance is negative and is clipped to zero. Consequently, the stored `heuristic_share = 0` must not be interpreted as evidence that the heuristics explain nothing: heuristic share is undefined when there is no positive explainable variance. The leave-one-out heuristic R² is −0.0228. Prompt bootstrap intervals remain broad: explainable fraction `[0, 0.430]`, cross-validated R² `[−0.125, 0.389]`, and cost range `[4.50%, 13.31%]`.

## Arm A

- High vs low reflection: −4.51 tokens (−0.90%), hierarchical 95% CI `[−25.74, 17.18]`; all seed differences favor high, but the interval crosses zero.
- High reflection vs random: −12.64 tokens (−2.48%), CI `[−32.04, 6.99]`; all seed differences favor high, but the interval crosses zero.
- Low reflection vs random: −8.13 tokens (−1.60%), CI `[−28.99, 10.57]`; seed signs are inconsistent.
- No accuracy contrast is significant.

Mean costs are 496.78 tokens for high reflection, 501.30 for low reflection, 509.42 for random, and 491.61 for the unadapted base. Relative to base, all three adapted conditions are slightly more expensive: +1.05%, +1.97%, and +3.62%, respectively.

## Scope and interpretation

The defensible claim is: **natural subset composition produced no detectable inference-cost effect beyond seed noise under this 15-step recipe.**

Claims that composition never matters, that heuristic features explain zero effect, or that the conditions are practically equivalent are not supported. Fourteen of sixteen Arm B subsets have only one seed, and the bootstrap upper bound still permits a material explainable fraction. Most token variation is carried by MATH-500; GSM8K is comparatively stable.

## Recommendation

Do not expand the existing 15-step matrix. If the research question concerns effects after meaningful dataset exposure, run a smaller, targeted dose-response replication at later checkpoints (for example 60 and 230 steps) using the most informative contrasts and at least three seeds per condition. Treat the current experiment as a well-executed low-dose null and use it to narrow, not close, the hypothesis.

Machine-readable evidence is in `docs/stage1c_matrix_audit.json` and `docs/stage1c_matrix_analysis.json`. The verifier is `scripts/audit_stage1c_matrix.py`; its server-side output is `results/stage1c_matrix/audit.json`.
