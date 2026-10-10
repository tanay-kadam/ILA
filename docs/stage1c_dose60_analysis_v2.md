# Stage 1c dose-60 analysis

Decision: STOP_QUALITY_FAILURE

Nine runs; raw metrics, evaluation alignment, adapter provenance, and training dose verified.

## armA_high_vs_armA_random

Mean token difference: +656.95 (+18.49%). Hierarchical 95% CI: [+388.10, +901.80].

Accuracy difference: -10.80 percentage points. Seed signs consistent: True. Step-230 gate: False.

Step-15 token difference: -12.64; change in contrast from step 15 to 60: +669.59 tokens (descriptive comparison).

## armA_low_vs_armA_random

Mean token difference: -751.03 (-21.14%). Hierarchical 95% CI: [-1012.47, -516.09].

Accuracy difference: +11.83 percentage points. Seed signs consistent: True. Step-230 gate: False.

Step-15 token difference: -8.13; change in contrast from step 15 to 60: -742.90 tokens (descriptive comparison).

Signals and quality failures are reported separately. Historical results are preserved; baseline checks added in v2 are retrospective for older runs.
