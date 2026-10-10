# Teacher-forced scalar-loss diagnostic: completed (2026-10-09)

Separate posthoc diagnostic, not retraining or new primary benchmark scores. Sources:
`tmlr_eos_effectiveness_probe_qwen.json` and `tmlr_eos_effectiveness_probe_mistral.json`.

Both families completed all21 arm/checkpoint conditions (base on three arm samples,
plus18 final adapters),32 fixed source-training examples per condition. In total:
42 condition summaries,1344 teacher-forced forwards, plus separate smoke forwards.
GPUs0/1 were selected through the allowlist/leases; peak allocated memory was28.72 GiB
for Qwen and25.64 GiB for Mistral, below the34 GiB bound. No optimizer or model-weight
write occurred. Human review labels remain empty.

Added objective share is R=9*sum(terminal EOS CE)/sum(native supervised-token CE).
Every condition falls below the frozen1% scalar-share screen:

- Qwen maximum R: **0.1264%** across its21 conditions.
- Mistral maximum R: **0.1681%** across its21 conditions.

This supports a small scalar teacher-forced terminal term on the sampled base/final
contexts. It qualifies interpretation of the negative fixed-weight10 experiment, but
does not establish the cause of that negative result. It does not measure historical
intermediate training gradients, gradient norms, dropout-active loss, optimizer update
influence, or free-running termination. A small scalar term may still influence gradients.
Numerically saturated probabilities/CE are not mathematical certainty or equivalence.

Keep this exploratory sample separate from the36-training/72-evaluation primary EOS
matrix. Do not use it to retune weights, redefine the10pp primary cap-reduction gate,
claim all termination mechanisms ineffective, or inflate benchmark evaluation counts.
