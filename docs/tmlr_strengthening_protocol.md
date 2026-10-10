# TMLR strengthening protocol — 2026-10-07

This is a prospective repository protocol, not public-registry preregistration. Prior
GSM8K/MATH experiments remain exploratory and are preserved. Negative or heterogeneous
findings will be reported; the following work does not presume acceptance or an effect.

## Questions and falsification

H1: The large composition-associated capped-cost gaps are predominantly associated with
changes in cap-hit frequency, rather than longer finished generations. Test an exact
descriptive decomposition and hierarchical bootstrap intervals. H1 is not supported if
the frequency component does not dominate the observed gap. This is an algebraic
description, not causal mediation: the finished cohorts can contain different problems.

H2: The ordering of capped cost, accuracy degradation, and repetition seen at high
adaptation strength generalizes to development-held-out math benchmarks. Failure to
reproduce, a stable large effect, or a reversed ordering narrows or falsifies that claim.

H3: The qualitative stability-versus-cost pattern is not unique to Qwen2.5-7B-Instruct.
Replicate in Mistral-7B-Instruct-v0.3. A divergent result is reported as a model-dependent
boundary, not hidden or followed by a search for a more favorable second model.

## A. Existing-output analysis

Use all three Arm A conditions and three seeds for the existing 1e-5/15-step,
1e-5/60-step, and 3e-6/60-step recipes. Analyze GSM8K and MATH separately and pooled.
Write separate artifacts; never overwrite original reports. Report capped mean, accuracy,
truncation, conservative loop rate, and tokens per correct answer with intervals.

For mean m, cap C, cap-hit probability q, and finished mean u:
m = q*C + (1-q)*u. For A-B, symmetric frequency component is
(q_A-q_B)*(C-(u_A+u_B)/2); finished-length component is
(u_A-u_B)*(1-(q_A+q_B)/2). Their sum is exactly the observed mean difference.
Report signed components, not a bounded causal percentage. If an arm has no finished
generations, report the symmetric decomposition as unidentified. Also report the exact
capped-row and uncapped-row contributions, cap sensitivity at 1,024/2,048/4,096/6,144
tokens, and descriptive common-finished-question comparisons with selection caveats.

Primary uncertainty retains independent seed resampling within arms plus shared prompt
resampling, 2,000 draws, seed 20261007. Paired-seed resampling is a labeled sensitivity
analysis, not a replacement chosen for significance. No equivalence claim is permitted
unless the entire relevant interval falls within a prospectively specified +/-5% margin;
wide inconclusive intervals are reported as inconclusive. This is not an exact TOST test.

## B. Frozen independent evaluation

Sources: authors' SVAMP repository (arkilpatel/SVAMP, SVAMP.json) and Google DeepMind's
AQuA repository (test.json). Archive the raw bytes, download URLs, revisions when available,
and SHA-256 hashes. Use no train/validation examples from those sources.

Exclude exact normalized-question and 13-word-gram overlaps with all prior natural
training-set questions and with the natural-study GSM8K-500/MATH-500 questions. Record exclusions, retain
the unfiltered source data, and inspect no model outcomes during filtering. Require at
least 300 retained SVAMP problems and 150 retained AQuA test problems; otherwise stop
data preparation for review, without substituting another benchmark based on results.
Select 400 SVAMP examples (or all retained if fewer) by fixed salted UID-hash order;
use all retained AQuA test examples. Freeze a manifest before generating any predictions.
SVAMP is numeric QA; AQuA is five-option letter QA with an explicit boxed-letter instruction.

These benchmarks are development-held-out for this study, not guaranteed absent from
foundation-model pretraining. Both remain mathematical domains, not broad cross-domain
generalization. Gold scoring is numeric for SVAMP and normalized exact option-letter
matching for AQuA. The latter requires a separate validated scorer.

Evaluate the base plus all nine Arm A checkpoints from each of the three existing Qwen
recipes: 28 checkpoints, each on both new sets, with canonical chat, greedy decoding,
6,144-token cap, and dynamic LoRA. Save generated token IDs, fingerprints, versions,
commands, device, raw predictions, and summaries. No new hyperparameter selection on
these outcomes, no cap increases, no hidden failed runs.

## C. Second-family replication

Freeze Mistral-7B-Instruct-v0.3 revision before downloading weights. Train high/low/random
at seeds 0/1/2 for two fixed LRs (1e-5 and 3e-6), planned cosine horizon 230, stop at 60,
and save step-15 snapshots for the 1e-5 trajectory. This requires 18 new training runs,
yielding the same three primary recipes and 27 evaluated adapters. Baseline is the
unadapted Mistral under the same prompt and scoring protocol.

Use the exact existing assigned subsets without outcome-dependent filtering. Canonical
Mistral serialization must pass prefix-mask and supervised-EOS checks. Use maximum
length 4,096, microbatch 1, accumulation 16, rank 64/alpha128/dropout .05 and the same
seven projection families. These model-specific resource settings are fixed for all
Mistral conditions; comparisons across model families are replication cases, not an
isolated causal architecture intervention. Precheck every training example for truncation;
any truncation or serialization mismatch stops the family branch for review.

Evaluate all 27 adapters plus base on the two frozen external sets. Baseline-relative
quality criteria remain accuracy loss <=5 percentage points per benchmark, truncation
<2%, and conservative loop rate <=2%. Composition contrasts are primary high-random and
low-random, with all seeds, hierarchical intervals, and a >=5% practical effect threshold.
Failure of quality, absence of signal, or model-dependent results are distinct outcomes.
No automated additional model, LR, or checkpoint search follows a failure.

## Resources and execution

GPU allowlist: 0,1,2,3,4,7. GPUs 5 and 6 are prohibited at every launch and worker check.
Do not kill or preempt another user's processes. Start on available allowlisted devices;
wait when sufficient free memory is unavailable. Bound GPU allocation per job and retain
failure logs. Run smoke tests and explicit device/config verification before full jobs.
Queue server work detached so a laptop VPN disconnect does not kill experiments.

Independent evaluation and second-family training have separate outputs/statuses, so a
blocked model download or failed smoke cannot erase or halt completed Qwen evidence.
Each branch ends with an artifact-integrity check and analysis, never automatic submission.

## Manuscripts and publication boundaries

Complete missing dataset methods from build_report.json, builder and feature code;
verify closest related papers; replace draft-status prose with a factual disclosure;
add seed- and benchmark-resolved plots and exact decompositions. Update a numerical
evidence packet and claim map. Draft/compile/review the TMLR manuscript with official
style files; provide human-review limitations. Workshop drafting remains subject to
the future non-archival track's human-authorship and submission policies.

Do not claim universal absence, proof of equivalence, causal identification of reflection,
a validated attribution estimator, independent pristine prior confirmation, or a paper
acceptance. No submission, external publication, or author-contact action is authorized
by this experiment controller.
