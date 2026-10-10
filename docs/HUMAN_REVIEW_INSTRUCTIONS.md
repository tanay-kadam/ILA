# Human validation: instructions and pending deliverable

Status: preparation does not count as validation. Only actual human decisions count.
Use `data/human_review_v1/reviewer_items.jsonl` and an EMPTY label template for each
reviewer. Do not share the private key, experiment analyses, or other reviewer's labels
before independent labeling. Do not ask an AI to fill the labels and call them human.

Enter a real reviewer identifier, one decision and a short reason per item. Preserve
item_id and task. Leave uncertain cases explicitly uncertain, not guessed. We recommend
two independently reviewing people; one-person review must be disclosed as such and
cannot produce inter-rater agreement. Keep the date and whether any AI assistance was
used in the reviewer notes. The visible output itself can reveal style clues, so blinding
is limited to removal of explicit condition/automatic-label/gold metadata.

## Loop assessment

Decision: `yes`, `no`, or `uncertain`.

Label yes if the output contains unproductive repeated reasoning (e.g. the same or
near-identical passage/steps repeated at least three times with no meaningful progress).
Normal one-off checking, alternative methods, repeated equations with new deductions,
or a long but productive solution are not alone loops. Judge the whole output, not
whether it reaches the correct answer. Cap status and the automatic loop flag are hidden.
Record uncertainty if the distinction cannot be made confidently.

## Final choice interpretation

Decision: `A`, `B`, `C`, `D`, `E`, `ambiguous`, or `unparseable`.

Identify the final intended option from the output and supplied option contents without
seeing gold. A clearly unique option value may identify an option. Conflicting final
letter/value or multiple unreconciled answers should be ambiguous. No identifiable final
answer should be unparseable. Do not choose a convenient intermediate number from the
reasoning. This task validates answer interpretation, not mathematical correctness of
the derivation. Explain which final phrase/box justified the decision.

## After labeling

Return the completed A/B JSONL label files separately. A later analysis must validate
IDs, coverage, allowed labels and reviewer identity, report missing/uncertain labels,
and compute agreement only when two genuine independent label sets exist. Discuss
disagreements after both independent files are frozen; retain pre-adjudication labels.
Loop rates/precision need the supplied stratification weights and uncertainty; the
purposefully selected option cases must not be used as an unbiased new benchmark score.
No empty template, automated mapping, or exported sample is evidence of human review.
