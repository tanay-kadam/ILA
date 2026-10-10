"""Export manuscript-ready numbers only after both fixed branches complete."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"results/tmlr_strengthening_evidence"
reports={family:json.loads((ROOT/f"results/tmlr_external_{family}/analysis.json").read_text()) for family in ("qwen","mistral")}
if any(report.get("integrity")!="PASS" for report in reports.values()):raise SystemExit("integrity not verified")
out.mkdir(exist_ok=True)
with open(out/"condition_metrics.csv","w",newline="") as f:
    writer=csv.writer(f);writer.writerow(["family","recipe","benchmark","condition","mean_tokens","accuracy","truncation_rate","loop_rate","quality_pass_all_seeds","source"])
    for family,report in reports.items():
        for recipe,sets in report["recipes"].items():
            for dataset,data in sets.items():
                for arm,metrics in data["arms"].items():
                    writer.writerow([family,recipe,dataset,arm,metrics["mean_tokens"],metrics["accuracy"],metrics["truncation_rate"],metrics["likely_loop_rate"],metrics["quality_pass_all_seeds"],f"results/tmlr_external_{family}/analysis.json"])
with open(out/"contrast_components.csv","w",newline="") as f:
    writer=csv.writer(f);writer.writerow(["family","recipe","benchmark","contrast","delta_tokens","relative_delta","ci95_low","ci95_high","frequency_component","finished_length_component","source"])
    for family,report in reports.items():
        for recipe,sets in report["recipes"].items():
            for dataset,data in sets.items():
                for contrast,c in data["contrasts"].items():
                    lo,hi=c["bootstrap"]["delta_tokens"]["ci95"]
                    writer.writerow([family,recipe,dataset,contrast,c["delta_tokens"],c["relative_delta"],lo,hi,c.get("frequency_component"),c.get("finished_length_component"),f"results/tmlr_external_{family}/analysis.json"])
manifest=json.loads((ROOT/"data/tmlr_external/manifest.json").read_text())
(out/"evidence.json").write_text(json.dumps({"datasets":manifest,"families":reports,"status":"READY_FOR_SCIENTIFIC_REVIEW_NOT_AUTOMATIC_SUBMISSION"},indent=2)+"\n")
print(f"Verified evidence exported to {out}")
