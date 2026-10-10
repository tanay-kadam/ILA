# B provisional AI review: current assessment

Both current AI files contain127 unique valid item labels. A/new-B raw concordance is
86/95 loops(90.5%) and27/32 answer interpretations(84.4%); nine loop/five choice cases
remain different. This is concordance, not accuracy, independent human validation or
population reliability. The five missing-options explanations in old B are absent in
new B; checked numeric-option and final-label/value conflict cases now use the options.
Do not assume all127 semantic decisions are correct or choose A simply because it is A.

Current files are explicitly AI/not-human. The owner reports the earlier Claude B was
given only items, the B template and instructions, without A labels. No provider logs,
complete rerun prompt/context handling or verified model identity are recorded here.
The corrected rerun follows rubric/input feedback; do not present increased agreement
as pristine confirmatory validation or keep rerunning until reviewers agree.

Next: inspect `data/human_review_v1/author_adjudication_AI_DISAGREEMENTS_PENDING.jsonl`.
It contains all14 disputed items with full question/options/output and both AI reasons.
`author_decision` is empty. Enter your own decision and a brief evidence-based reason;
retain uncertainty instead of forcing consensus. Primary experiment scores stay frozen.

For choice interpretation, use the supplied options. A final numeric value uniquely
matching an option can identify it. Distinguish conflicting final label/value from merely
incorrect earlier reasoning; use the same precedence rule across all cases. For loops,
distinguish bounded useful enumeration from unproductive repetition/restarted scans.
Changing numbers or reaching an answer alone does not settle the question.

This would be single-author AI-assisted targeted adjudication, not independent blinded
validation of127 items. Agreement on the other113 items is not proof they are correct;
a separate fixed sample of agreed cases can be checked if broader validation is desired.
The stratified/purposeful sampling limitations still apply. Do not report these AI labels
through `analyze_human_review.py` with a false human-review attestation.

The older `reviewer_B_labels_AI_REVIEW.jsonl` is no longer present in the current worktree.
Its earlier audit/hash/concordance history is retained, but its original raw reasons
cannot now be re-read from that path. No old label file has been reconstructed or
silently overwritten. Keep the current provisional file and any genuine original backup.
