"""Separate composition signals from model-quality failures."""

def contrast_gate(result, accuracy_loss, max_truncation, baseline_pass=True):
    lo, hi = result["ci95_hierarchical"]
    ci_signal = (lo > 0 or hi < 0) and abs(result["rel_delta_tokens"]) >= 0.02
    large_signal = abs(result["rel_delta_tokens"]) >= 0.05 and result["sign_consistent_across_seeds"]
    signal = bool(ci_signal or large_signal)
    quality = bool(accuracy_loss <= 0.01 and max_truncation < 0.02 and baseline_pass)
    return {"ci_signal": bool(ci_signal), "large_consistent_signal": bool(large_signal),
            "signal_gate_pass": signal, "guardrails_pass": quality,
            "preferred_accuracy_loss": float(accuracy_loss), "max_truncation_rate": float(max_truncation),
            "baseline_guardrails_pass": bool(baseline_pass), "continue_to_step230": signal and quality}


def outcome(gates):
    gates = list(gates)
    if not all(g["guardrails_pass"] for g in gates):
        return "STOP_QUALITY_FAILURE"
    if any(g["signal_gate_pass"] for g in gates):
        return "STABLE_SIGNAL_REVIEW"
    return "STOP_NO_SIGNAL"
