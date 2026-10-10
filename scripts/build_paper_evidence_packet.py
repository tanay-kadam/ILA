"""Bundle existing numerical artifacts and methods for grounded manuscript drafting."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_FILES = [
    "results/stage1c_matrix/analysis.json",
    "results/stage1c_matrix/audit.json",
    "results/stage1c_dose60/analysis.json",
    "results/stage1c_dose60/analysis_v2.json",
    "results/stage1c_dose60/independent_audit/audit.json",
    "results/stage1c_dose60/independent_audit/hf_backend_full.json",
    "results/stage1c_dose60/independent_audit/tokenizer_train.json",
    "results/stage1c_dose60/independent_audit/tokenizer_eval.json",
    "results/stage1c_final_stability/final_report.json",
    "results/stage1c_q7_lr1e5/engineering_gate.json",
    "data/stage1/build_report.json",
    "results/tmlr_existing_analysis/analysis.json",
    "data/tmlr_external/manifest.json",
    "docs/tmlr_mistral_precheck.json",
    *[f"results/{name}/analysis.json" for name in ("gate0", "gate0_para", "gate0_1p5b", "gate0_para_1p5b")],
]
TEXT_FILES = [
    "docs/stage1c_matrix_execution.md", "docs/stage1c_dose_response.md",
    "docs/stage1c_final_stability.md", "docs/stage1c_dose60_pipeline_audit.md",
    "docs/stage1c_audit_fixes_and_rerun_policy.md",
    "docs/tmlr_strengthening_protocol.md", "docs/tmlr_dataset_methods.md",
    "configs/stage1.yaml", "configs/stage1b_q7.yaml", "configs/stage1c_q7.yaml",
    "configs/stage1c_q7_lr1e5.yaml", "configs/stage1c_matrix.yaml",
    "configs/stage1c_dose60.yaml", "configs/stage1c_final_stability.yaml",
    "scripts/analyze_gate0.py", "scripts/analyze_stage1.py",
    "scripts/analyze_stage1c_dose60.py", "scripts/gate_final_stability.py",
    "scripts/diagnose_stage1_outputs.py", "src/ila/dose_gate.py",
    "scripts/build_natural.py", "src/ila/natural.py", "src/ila/features.py",
    "src/ila/cost_boundary.py", "scripts/analyze_tmlr_existing.py",
]
packet = {"project": "Inference Liability Attribution", "snapshot_date": "2026-10-07",
          "purpose": "Evidence for research drafts, not certification of publication readiness",
          "numeric_artifacts": {}, "methods_and_notes": {}, "missing_files": []}
for relative in JSON_FILES + TEXT_FILES:
    path = ROOT / relative
    if not path.exists():
        packet["missing_files"].append(relative)
        continue
    raw = path.read_bytes()
    entry = {"sha256": hashlib.sha256(raw).hexdigest(), "content": json.loads(raw) if relative in JSON_FILES else raw.decode("utf-8")}
    packet["numeric_artifacts" if relative in JSON_FILES else "methods_and_notes"][relative] = entry
out = ROOT / "docs/paper_evidence_packet.json"
out.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "numeric_artifacts": len(packet["numeric_artifacts"]),
                  "methods_and_notes": len(packet["methods_and_notes"]), "missing": packet["missing_files"]}))
