# Natural training-data methods for manuscript revision

Verified from `scripts/build_natural.py`, `src/ila/natural.py`, `src/ila/features.py`,
the resolved configs and `data/stage1/build_report.json`. This is a methods source
note, not a claim that every source trace was manually checked.

The source is `open-r1/OpenR1-Math-220k`, configuration `default`, split `train`.
The recorded build processed 93,733 source rows. For each problem it selected the
first supplied generation whose dataset flags indicated mathematical correctness,
completed reasoning, and (when supplied) finish reason `stop`. These are source
annotations, not independent human verification by this study.

The build rejected questions overlapping the MATH-500 problems or GSM8K test questions
under normalized 13-word grams (and exact normalized short questions). It retained
traces with 1,000–2,500 Qwen-token target length and at most 400 problem tokens, yielding
an 11,491-trace candidate pool. The recorded exact training/evaluation question overlap
is zero. This screen reduces overlap; it does not establish absence of semantic
near-duplicates or foundation-model pretraining contamination.

Reflection density is 1,000 times the count of case-insensitive regex matches divided
by target token count. Patterns cover `wait`, `let me (re)check`, `let me (re)verify`,
`double-check`/`double check`, `alternatively`, and `hmm`. This is a surface-marker
heuristic, not a semantic annotation of reasoning. Restart and boxed-answer counts
are additional inexpensive features.

Arm A partitions the pool by target-length decile, source, and problem type. Eligible
bins have at least six examples; per-bin allocations are shared across all three
conditions. The high and low subsets take reflection-density extremes, while the
random control samples uniformly without replacement within each bin. The three
subsets have 3,676 traces each and identical bin quotas; 170 of 310 bins contribute.
Random can overlap high or low. High/low differ in their underlying questions and
other trace characteristics; matching does not uniquely identify reflection causality.

Measured mean target lengths are high 1,864.458, random 1,867.013, and low 1,869.058
tokens. Mean reflection densities are high 6.7943, random 4.5883, low 2.5887. The
high-versus-low relative mean length gap is -0.2461%. The source mix is approximately
83.05% olympiads, 11.48% Chinese contest problems, 4.52% AoPS forum, and 0.95% AMC/AIME
within each Arm A condition. This concentration limits domain generalization.

Arm B applies TF-IDF to pool questions (20,000 maximum features, English stop words,
sublinear TF) and k-means with 16 clusters, three initializations, and a fixed selection
seed. Each of 16 subsets draws a cluster mixture from a symmetric Dirichlet with
concentration .5, then uses capped integer apportionment and within-cluster sampling
without replacement to obtain 4,000 examples. Subsets may overlap. Report realized
mixtures, not just intended Dirichlet draws, and do not imply independent training sets.

The current fresh-evaluation screen in `prepare_tmlr_external.py` is against all
natural-training files in `data/stage1` and the previously evaluated natural-study
GSM8K-500/MATH-500 questions. It is not a claim of exhaustive project-wide or
foundation-pretraining decontamination. Both SVAMP and AQuA are newly used benchmarks
for this study, but both are still mathematical QA tasks.
