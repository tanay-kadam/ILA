"""Build source-hashed ChatGPT handoff; never edit manuscripts or infer run status."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    registry_path = ROOT / "docs/experiment_registry.json"
    ledger = json.loads(registry_path.read_text(encoding="utf-8"))
    paths = sorted({p for e in ledger["experiments"] for p in e.get("sources", [])})
    artifacts = {}
    for name in paths:
        path = ROOT / name
        raw = path.read_bytes()
        entry = {"local_path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        if path.suffix == ".json":
            entry["content"] = json.loads(raw)
        elif path.suffix in (".md", ".py"):
            entry["content"] = raw.decode("utf-8")
        else:
            entry["content_included"] = False
            entry["note"] = "Non-JSON/Markdown source available separately. Case exports are NOT human reviewed."
        artifacts[name] = entry
    for family in ("qwen", "mistral"):
        for kind in ("external", "format_control"):
            name = f"docs/tmlr_{kind}_{family}_analysis.json"
            if artifacts[name]["content"].get("integrity") != "PASS":
                raise ValueError(f"missing integrity PASS: {name}")
    comparison = artifacts["docs/tmlr_format_control_comparison.json"]["content"]
    changes = {}
    for family, result in comparison["families"].items():
        original, controlled = result["baseline_original"], result["baseline_control"]
        changes[family] = {
            "baseline_original": original, "baseline_format_control": controlled,
            "accuracy_change_percentage_points": 100*(controlled["accuracy"]-original["accuracy"]),
            "valid_letter_change_percentage_points": result["baseline_letter_rate_gain_pp"],
            "control_valid_letter_ge_90percent": result["baseline_letter_rate_ge_90percent"],
            "contrast_comparison": result["contrast_comparison"],
            "source": "docs/tmlr_format_control_comparison.json"}
    claims = [
        {"claim": "Mistral formatting improved by the prespecified >=20 percentage-point threshold.", "classification": "SUPPORTED WITH CAVEAT", "source": "docs/tmlr_format_control_comparison.json / hypothesis and families/mistral", "caveat": "Valid-letter rate improves from 24.41% to 79.13%, still below 90%. Exploratory reused-benchmark control, not universally good mathematical reasoning."},
        {"claim": "All composition-cost gaps are robust to formatting prompts.", "classification": "UNSUPPORTED", "source": "docs/tmlr_format_control_comparison.json / families/qwen/contrast_comparison/lr1e5_step60/armA_high_vs_random", "caveat": "AQuA high-random changes from +743.59 to -12.84 tokens; control CI spans zero. Not proof of zero effect, equivalence, or a formally tested interaction."},
        {"claim": "Mistral large 60-step cost contrasts persist under this fixed formatting prompt.", "classification": "SUPPORTED WITH CAVEAT", "source": "docs/tmlr_format_control_comparison.json / families/mistral/contrast_comparison", "caveat": "Limited to these recipes, AQuA and prompt package. Descriptive cost decomposition, not causal mediation or usable efficiency."},
        {"claim": "This prompt is an isolated syntax-only repair.", "classification": "OVERSTATED", "source": "docs/tmlr_format_control_comparison.json / families/qwen/baseline_original and baseline_control", "caveat": "Full prompt package changes generated behavior; Qwen base accuracy decreases from 83.86% to 78.35%."},
        {"claim": "The fixed terminal-EOS weight10 intervention fixes degradation.", "classification": "UNSUPPORTED", "source": "docs/tmlr_eos_control_comparison.json / families/{qwen,mistral}/primary_gate", "caveat": "Completed matched control did not meet the prespecified >=10pp cap-reduction criterion in either family; CIs span zero. This does not prove equivalence or rule out broader termination mechanisms. The separate frozen-checkpoint scalar diagnostic is completed, but historical training gradients were not logged."},
        {"claim": "Reflection independently causes the gap after controlling questions, length and training dose.", "classification": "UNVERIFIED", "source": "docs/experiment_registry.json / matched_trace_length_and_training_dose_control", "caveat": "Length/source/type matching and token-exposure balance were verified, but the strict exact-question design yields9 pairs of960 required and was not run. Retain composition-associated language, not causal reflection."}
    ]
    packet = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status_verified_as_of": ledger["last_verified_date"],
        "purpose": "Self-contained evidence handoff for human-led manuscript updates",
        "reading_order": ["experiment_registry", "interpretation_guardrails", "latest_verified_changes", "claim_review", "source_artifacts"],
        "experiment_registry": ledger,
        "interpretation_guardrails": [
            "Current registry statuses supersede old queued/in-progress prose in historical packets/protocols, not preserved numerical results.",
            "Separate primary strict accuracy, posthoc gold-blind content mapping, and new prompt-control results. Never silently substitute scores.",
            "Planned work has no results, compute duration or human annotations. Packet generation does not refresh live SSH status.",
            "New format control: 56 checkpoint evaluations, AQuA only, 254 rows, zero new training. Original external study: 400 SVAMP and 254 AQuA per checkpoint.",
            "Preserve family/recipe/benchmark/seed labels, capped generated-token definition, input overhead, uncertainty and quality failures.",
            "An interval spanning zero is inconclusive, not proof of equivalence. Different tokenizers/resource settings confound cross-family causal comparisons.",
            "Cap-frequency decomposition is descriptive, not causal mediation. Clipped old lengths are not actual larger-cap generation.",
            "No demonstrated causal reflection, validated attribution estimator, pretraining decontamination, acceptance probability or submission readiness.",
            "Recorded pipeline integrity checks and preliminary claim review are not independent publication-final verification.",
            "Read the current EOS audit: weighted_terminal_examples is dataset eligibility, not actual60-step exposure; missing historical gradient diagnostics limit null-mechanism conclusions. Pooled mechanism support is not per-seed usable quality.",
            "Matching feasibility is not a new model experiment: existing token/source/type balance passes its frozen screen, strict same-question design is infeasible. Blinded127-item bundle has zero completed human labels; prepared files do not establish validation."
        ],
        "latest_verified_changes": changes, "claim_review": claims, "source_artifacts": artifacts,
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    rendered = json.dumps(packet, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    target = ROOT / "docs/experiment_updates.json"
    target.write_text(rendered, encoding="utf-8")
    check = json.loads(target.read_text(encoding="utf-8"))
    assert len(check["source_artifacts"]) == len(paths)
    print(f"Wrote {target}: {target.stat().st_size:,} bytes; {len(paths)} artifacts; {len(ledger['experiments'])} experiment records")


if __name__ == "__main__":
    main()
