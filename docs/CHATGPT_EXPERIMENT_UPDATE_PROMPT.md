# Update manuscripts from the current experiment record

Upload **experiment_updates.json** from this directory and your current manuscript
sources (`.tex`, bibliography and style files; PDFs are useful companions).
The JSON includes historical evidence, completed external replication, full prompt-control
comparisons/per-seed analyses, audits, protocols, configs/hashes, statuses and caveats.
Do not send only this prompt and expect ChatGPT to know the numbers.

Paste the following into ChatGPT:

---

You are assisting with my human-led research manuscript updates. Use the attached
`experiment_updates.json` as the experimental/numerical source. Do not invent experiments,
numbers, citations, validation, authorship details, or results for planned work.

Read the packet in its specified order. Current registry statuses supersede outdated
queued/in-progress prose in historical source artifacts, not original numeric results.
Identify completed, partial, failed and not-started work before editing. Check
`status_verified_as_of`; do not imply that the packet describes live server state later.

First give a concise change plan and identify claims to narrow/remove. Then update my
supplied manuscripts, preserving structure and venue scope unless scientific corrections
require changes. If only PDFs are supplied, explain reconstruction limitations. Return
complete updated `.tex` files where feasible, a concise change log, and a claim-to-evidence
map with exact artifact paths/JSON keys. Preserve verified historical numbers unless an
explicit correction supports changing them; flag conflicts instead of picking the stronger
story. I remain responsible for scientific review and writing.

Mandatory constraints:

- Separate original strict boxed-letter accuracy, posthoc gold-blind option-content
  diagnostics, and new fixed-prompt generation. No silent score replacements.
- Include all relevant families, recipes, benchmarks, arms and seeds, including null,
  reversed, degraded and prompt-sensitive results. Use recorded CIs/quality gates.
- The prompt control contains 56 evaluations on 254 AQuA problems, zero new training.
  Original external evaluations use 400 SVAMP and 254 AQuA problems per checkpoint.
- Mistral valid-letter rate improves 24.41% to 79.13%, still below 90%. Strict base
  accuracy improves 10.63% to 28.74%; this is not independent proof of improved reasoning.
- Qwen base accuracy falls 83.86% to 78.35%. Its 1e-5/60-step AQuA high-random capped-token
  gap changes +743.59 to -12.84, with control CI spanning zero. Do not hide these results,
  claim equivalence, or invent a tested prompt-by-arm interaction. Qwen low-random and
  Mistral large 60-step gaps persist; use exact values/intervals from the packet.
- Cost is capped generated tokens, not uncensored expected cost or usable efficiency.
  Cap-frequency decomposition is descriptive, not causal mediation. Finished cohorts,
  model tokenizers and family training resource differences constrain comparisons.
- Terminal-EOS control completed all36 matched weight1/10 trainings and72 evaluations,
  zero failed jobs; automatic analysis passed. Both families failed the frozen >=10pp
  cap-reduction criterion and their cap-difference CIs span zero. Use exact results from
  tmlr_eos_control_comparison.json in the packet, not smoke metrics. Do not conclude
  equivalence or rule out broader termination mechanisms from this fixed intervention.
  Final raw/fingerprint verification passed. Original arms already match length/source/
  type bins; canonical and verified-sampler token exposure are within declared margins.
  The strict same-question design is infeasible under frozen criteria(9 pairs vs960
  required); no new matched GPU experiment was run. Do not treat this as causal evidence.
  The separate completed frozen-checkpoint diagnostic has42 condition summaries,32
  training cases per condition; added scalar EOS CE share is<0.17% in all conditions.
  This is posthoc teacher forcing, not historical gradient measurement or a new primary
  benchmark score. Do not infer small gradient influence or all-mechanism ineffectiveness.
  Blinded127-item review bundle is prepared but actual human labels remain absent.
  Longer-cap generation is not performed. Exported cases/empty templates are not labels.
- Keep exploratory/posthoc and reused-benchmark disclosures, seed/power uncertainty,
  contamination caveats and confounds. No causal reflection, validated attribution
  estimator, universal prompt robustness, guaranteed publication or acceptance probability.

Use the preliminary claim review as guidance, not a substitute for consistency checks.
Verify new paper numbers against source artifacts and aggregation/formulas. Missing
metrics must remain missing. Do not fabricate p-values, interactions, equivalence tests,
human agreement or CIs. Do not submit anything. If discussing current venue policies,
deadlines, archival overlap or AI-authorship rules, verify official sources separately;
the experiment packet is not a policy reference.

---

## Maintenance by the experimentation agent

After each launch/result review, update `experiment_registry.json` with verified status,
dates, sources and failures; collect new analyses without weights; update the builder's
claim caveats if needed; regenerate with `python scripts/build_experiment_update_packet.py`.
The builder does not poll SSH, launch runs or make stale evidence current by itself.
Do not edit manuscripts or original scores. Keep failed evidence. The older
`paper_writing_prompt.md` is historical; use this prompt for current updates.
