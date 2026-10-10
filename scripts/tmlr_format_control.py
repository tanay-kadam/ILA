"""Fixed AQuA prompt control: precheck -> leased smokes -> queue -> analysis.

Run detached on the server. Does not train, select checkpoints, or edit manuscripts.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ila.config import load_config
from ila.gpu_lease import gpu_lease
from ila.textio import read_jsonl, build_chat_user_content, build_prompt, extract_boxed, answers_match_choice
from evaluate import file_hash
from tmlr_gpu_worker import ALLOWED, ready, register

CONTROL = ROOT / "results/tmlr_format_control_controller"
TRAIN_PY = ROOT / ".venv/bin/python"
EVAL_PY = ROOT / ".venv-vllm/bin/python"
EXAMPLES = r"""Formatting examples (illustrations only, not the question to solve):
Example 1: What is 1 + 1? A)2 B)3 C)4 D)5 E)6.
Reasoning: 1 + 1 = 2, which is option A. Final answer: \boxed{{A}}.
Example 2: What is 9 - 4? A)1 B)2 C)4 D)5 E)6.
Reasoning: 9 - 4 = 5, which is option D. Final answer: \boxed{{D}}.

"""
REMINDER = r"""
The final boxed answer must contain exactly one option letter (A, B, C, D, or E),
not the numerical value or text of the option. Reason step by step before the final box.
"""


def control_template(original: str) -> str:
    # Insert before the trailing plain Answer cue, which canonical chat removes.
    import re
    original = re.sub(r"\nAnswer:\s*$", "", original).rstrip()
    return EXAMPLES + original + "\n" + REMINDER + "\nAnswer:\n"


def status(value):
    (CONTROL / "STATUS").write_text(value + " " + time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    print(value, flush=True)


def prepare():
    if CONTROL.exists() or any((ROOT / f"results/tmlr_format_control_{f}").exists() for f in ("qwen", "mistral")):
        raise RuntimeError("control output already exists; inspect before any restart")
    CONTROL.mkdir(parents=True)
    status("PRECHECK")
    rows = read_jsonl(ROOT / "data/tmlr_external/eval_aqua.jsonl")
    if len(rows) != 254 or len({r["uid"] for r in rows}) != 254:
        raise ValueError("frozen AQuA size/UID mismatch")
    inventory = {"protocol_sha256": file_hash(ROOT / "docs/tmlr_format_control_protocol.md"),
                 "data_sha256": file_hash(ROOT / "data/tmlr_external/eval_aqua.jsonl"),
                 "manifest_sha256": file_hash(ROOT / "data/tmlr_external/manifest.json"),
                 "gpu_allowlist": sorted(ALLOWED), "families": {}, "adapter_sha256": {}}
    for family in ("qwen", "mistral"):
        original_path = ROOT / f"configs/tmlr_external_{family}{'_resolved' if family == 'mistral' else ''}.yaml"
        original = load_config(original_path)
        cfg = copy.deepcopy(original)
        cfg["experiment"] = f"tmlr_aqua_fixed_format_control_{family}"
        cfg["results_dir"] = f"results/tmlr_format_control_{family}"
        cfg["prompt_template"] = control_template(original["prompt_template"])
        for row in rows:
            # Validate brace escaping before creating any GPU job.
            build_prompt(row["question"], cfg["prompt_template"])
        config_path = ROOT / f"configs/tmlr_format_control_{family}_resolved.yaml"
        if config_path.exists():
            raise RuntimeError(f"config already exists: {config_path}")
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        jobs = [{"id": f"{family}_base", "mode": "eval", "config": str(config_path),
                 "style": "_base", "seed": 0, "base": True, "eval_sets": ["aqua"]}]
        original_root = ROOT / f"results/tmlr_external_{family}"
        for recipe in ("lr1e5_step15", "lr1e5_step60", "lr3e6_step60"):
            for arm in ("armA_high", "armA_low", "armA_random"):
                for seed in range(3):
                    style = f"{recipe}__{arm}"
                    rec_path = original_root / style / f"seed{seed}/run_record.json"
                    rec = json.loads(rec_path.read_text())
                    if rec["status"] != "completed" or rec["model_dir"] != cfg["model"] or rec["seed"] != seed:
                        raise ValueError(f"source record mismatch: {rec_path}")
                    adapter = Path(rec["adapter_dir"])
                    weights = sorted(list(adapter.glob("*.safetensors")) + list(adapter.glob("*.bin")))
                    if not weights or not (adapter / "adapter_config.json").is_file():
                        raise ValueError(f"adapter missing: {adapter}")
                    for path in weights + [adapter / "adapter_config.json"]:
                        if str(path) not in inventory["adapter_sha256"]:
                            inventory["adapter_sha256"][str(path)] = file_hash(path)
                    jobs.append({"id": f"{family}_{style}_seed{seed}", "mode": "eval", "config": str(config_path),
                        "style": style, "seed": seed, "adapter": str(adapter), "source": str(rec_path),
                        "step": rec["optimizer_step"], "eval_sets": ["aqua"]})
        # Existing base and all adapters must have complete original predictions.
        for job in jobs:
            summary = original_root / job["style"] / f"seed{job['seed']}/eval/aqua/summary.json"
            info = json.loads(summary.read_text())
            if info["n"] != 254 or info["limit"] is not None or info["max_new_tokens"] != 6144 or not info.get("evaluation_fingerprint"):
                raise ValueError(f"original evaluation incomplete: {summary}")
        queue = CONTROL / f"{family}_queue"
        queue.mkdir()
        (queue / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        lengths = {}
        for label, template in (("original", original["prompt_template"]), ("control", cfg["prompt_template"])):
            contents = [build_chat_user_content(r["question"], template) for r in rows]
            lengths[label] = [len(tok.apply_chat_template([{"role": "user", "content": c}], tokenize=True, add_generation_prompt=True)) for c in contents]
        if max(lengths["control"]) > 2048:
            raise ValueError("controlled prompt exceeds evaluator's input context allowance")
        inventory["families"][family] = {"config": cfg, "config_sha256": file_hash(config_path),
            "original_config": original, "input_token_counts": lengths, "jobs": len(jobs)}
    inventory["code_sha256"] = {str(p.relative_to(ROOT)): file_hash(p) for p in (
        ROOT / "scripts/evaluate.py", ROOT / "scripts/tmlr_format_control.py",
        ROOT / "scripts/tmlr_gpu_worker.py", ROOT / "scripts/analyze_tmlr_external.py", ROOT / "src/ila/choice_audit.py")}
    (CONTROL / "precheck.json").write_text(json.dumps(inventory, indent=2) + "\n")
    status("PRECHECK_PASSED")


def run_smoke(family, job):
    while True:
        lease = None
        for gpu in sorted(ALLOWED):
            if not ready(gpu):
                continue
            candidate = gpu_lease(ROOT / "results/tmlr_strengthening_controller/gpu_leases", gpu)
            if candidate.__enter__():
                if ready(gpu):
                    lease = candidate
                    break
                candidate.__exit__(None, None, None)
            else:
                candidate.__exit__(None, None, None)
        if lease is not None:
            break
        print("WAIT_SMOKE_CAPACITY allowlist=0,1,2,3,4,7", flush=True)
        time.sleep(30)
    try:
        cfg = load_config(job["config"])
        cfg["results_dir"] = f"results/tmlr_format_control_smoke_{family}"
        base = job.get("base", False)
        style = "_base" if base else "adapter_smoke"
        if not base:
            register(cfg, style, job["seed"], job["adapter"], job["source"], job["step"])
        command = [str(EVAL_PY), "scripts/evaluate.py", "--config", job["config"], "--style", style,
            "--seed", str(job["seed"]), "--eval-set", "aqua", "--limit", "2", "--set", f"results_dir={cfg['results_dir']}"]
        if base:
            command.append("--base")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "PYTHONPATH": "src",
               "ILA_VLLM_UTIL": "0.70", "ILA_GPU_MEM_GB": "34"}
        log = CONTROL / f"smoke_{family}_{style}.log"
        with log.open("w") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        path = ROOT / cfg["results_dir"] / style / f"seed{job['seed']}/eval/aqua"
        summary = json.loads((path / "summary.json").read_text())
        rows = read_jsonl(path / "outputs.jsonl")
        frozen = read_jsonl(ROOT / "data/tmlr_external/eval_aqua.jsonl")[:2]
        if summary["n"] != 2 or summary["limit"] != 2 or summary["backend"] != "vllm" or summary["max_new_tokens"] != 6144:
            raise ValueError("smoke config mismatch")
        if [(r["uid"], r["gold"]) for r in rows] != [(r["uid"], r["gold"]) for r in frozen]:
            raise ValueError("smoke UID mismatch")
        for row in rows:
            if len(row["generated_token_ids"]) != row["n_gen_tokens"] or answers_match_choice(extract_boxed(row["output"]), row["gold"]) != row["correct"]:
                raise ValueError("smoke raw integrity mismatch")
        # This vLLM installation supports only GPU execution; require explicit log evidence.
        text = log.read_text()
        if "cuda" not in text.lower() or "Initializing a V1 LLM engine" not in text:
            raise ValueError("missing explicit vLLM CUDA engine evidence")
        record = {"status": "passed", "command": command, "resolved_config": cfg,
                  "physical_gpu": gpu, "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                  "summary": summary, "log": str(log)}
        (CONTROL / f"smoke_{family}_{style}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"SMOKE_PASSED family={family} style={style} physical_gpu={gpu}", flush=True)
    finally:
        lease.__exit__(None, None, None)


def compare():
    result = {"selection": "all fixed checkpoints; no outcome-based exclusions", "families": {}}
    for family in ("qwen", "mistral"):
        original = json.loads((ROOT / f"results/tmlr_external_{family}/analysis.json").read_text())
        controlled = json.loads((ROOT / f"results/tmlr_format_control_{family}/analysis.json").read_text())
        old = original["baseline"]["aqua"]
        new = controlled["baseline"]["aqua"]
        gain = new["valid_boxed_option_letter_rate"] - old["valid_boxed_option_letter_rate"]
        family_report = {"baseline_original": old, "baseline_control": new,
            "baseline_letter_rate_gain_pp": 100 * gain, "baseline_gain_ge_20pp": gain >= .20,
            "baseline_letter_rate_ge_90percent": new["valid_boxed_option_letter_rate"] >= .90,
            "per_seed_differences": {}, "contrast_comparison": {}}
        keys = ("accuracy", "mean_tokens", "truncation_rate", "likely_loop_rate", "valid_boxed_option_letter_rate")
        for recipe in controlled["recipes"]:
            before = original["recipes"][recipe]["aqua"]
            after = controlled["recipes"][recipe]["aqua"]
            for arm in after["arms"]:
                family_report["per_seed_differences"][f"{recipe}__{arm}"] = [
                    {k: a[k] - b[k] for k in keys}
                    for a, b in zip(after["arms"][arm]["per_seed"], before["arms"][arm]["per_seed"])]
            family_report["contrast_comparison"][recipe] = {
                arm: {"original": before["contrasts"][arm], "control": after["contrasts"][arm]}
                for arm in after["contrasts"]}
        result["families"][family] = family_report
    result["hypothesis"] = {"test": "Mistral baseline boxed-letter rate gain >=20 percentage points",
                            "supported": result["families"]["mistral"]["baseline_gain_ge_20pp"]}
    (CONTROL / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.chdir(ROOT)
    os.environ.update(HF_HUB_OFFLINE="1", PYTHONPATH="src")
    # Prepare refuses an existing directory. Never overwrite its status on failed restart.
    if CONTROL.exists():
        raise SystemExit("controller exists; preserved without restart")
    try:
        prepare()
        for family in ("qwen", "mistral"):
            jobs = json.loads((CONTROL / f"{family}_queue/jobs.json").read_text())
            status(f"SMOKE_{family.upper()}")
            run_smoke(family, jobs[0])
            run_smoke(family, jobs[1])
        status("SMOKES_PASSED_RUNNING_56_CHECKPOINT_QUEUE")
        workers = []
        streams = []
        for family in ("qwen", "mistral"):
            queue = CONTROL / f"{family}_queue"
            for gpu in sorted(ALLOWED):
                command = [str(TRAIN_PY), "scripts/tmlr_gpu_worker.py", "--jobs", str(queue / "jobs.json"), "--queue", str(queue), "--gpu", str(gpu)]
                stream = (queue / f"worker_gpu{gpu}.log").open("w")
                streams.append(stream)
                workers.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
        codes = [worker.wait() for worker in workers]
        for stream in streams:
            stream.close()
        if any(codes):
            raise RuntimeError(f"worker exit codes: {codes}")
        for family in ("qwen", "mistral"):
            queue = CONTROL / f"{family}_queue"
            jobs = json.loads((queue / "jobs.json").read_text())
            expected = {j["id"] for j in jobs}
            if list((queue / "failed").glob("*.json")) or {p.stem for p in (queue / "done").glob("*.json")} != expected:
                raise RuntimeError(f"{family}: failed or incomplete matrix; inspect retained logs")
        status("FULL_MATRIX_COMPLETED_RUNNING_INTEGRITY_AND_ANALYSIS")
        for family in ("qwen", "mistral"):
            output = f"results/tmlr_format_control_{family}"
            with (CONTROL / f"{family}_analysis.log").open("w") as stream:
                subprocess.run([str(TRAIN_PY), "scripts/analyze_tmlr_external.py", "--family", family, "--root", output, "--eval-sets", "aqua"], stdout=stream, stderr=subprocess.STDOUT, check=True)
            subprocess.run([str(TRAIN_PY), "scripts/audit_tmlr_aqua_format.py", "--root", output, "--output", str(CONTROL / f"{family}_content_audit.json")], check=True)
        compare()
        status("COMPLETED_VERIFIED_FORMAT_CONTROL_ANALYSIS_READY")
    except Exception as error:
        if CONTROL.exists():
            (CONTROL / "failure.json").write_text(json.dumps({"error": repr(error)}, indent=2))
            status("FAILED_RETAINED_ARTIFACTS_INSPECT_LOG")
        raise


if __name__ == "__main__":
    main()
