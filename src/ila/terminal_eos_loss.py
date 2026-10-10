"""Add terminal-only CE to native causal loss without changing its denominator."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def terminal_eos_extra(logits, labels, eos_token_id, num_items_in_batch=None, allowed_suffix_ids=()):
    valid = labels.ne(-100)
    positions = torch.arange(labels.shape[1], device=labels.device).expand_as(labels)
    last = positions.masked_fill(labels.ne(eos_token_id) | ~valid, -1).max(dim=1).values
    batch = torch.arange(labels.shape[0], device=labels.device)
    if torch.any(last < 1) or torch.any(labels[batch, last.clamp_min(0)] != eos_token_id):
        raise ValueError("each example must end in a supervised terminal EOS with a preceding prediction position")
    tail = valid & (positions > last[:,None])
    permitted = torch.zeros_like(valid)
    for token in allowed_suffix_ids:
        permitted |= labels.eq(token)
    if torch.any(tail & ~permitted):
        raise ValueError("unexpected supervised content after terminal EOS")
    denominator = num_items_in_batch if num_items_in_batch is not None else labels[:, 1:].ne(-100).sum()
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.to(logits.device)
    if denominator <= 0:
        raise ValueError("nonpositive supervised-token denominator")
    prediction = logits[batch.to(logits.device), (last-1).to(logits.device)].float()
    targets = labels[batch, last].to(logits.device)
    return F.cross_entropy(prediction, targets, reduction="sum") / denominator


def validate_terminal_dataset(dataset, eos_token_id, tokenizer=None, suffix_ids=None):
    if dataset.n_truncated:
        raise ValueError("EOS control refuses truncated training examples")
    for item in dataset.items:
        labels = item["labels"]
        valid = [i for i, label in enumerate(labels) if label != -100]
        eos = [i for i in valid if labels[i] == eos_token_id]
        if not valid or valid[0] == 0 or valid[-1] != len(labels)-1 or not eos:
            raise ValueError("invalid prompt mask or terminal EOS")
        if any(label == -100 for label in labels[valid[0]:]):
            raise ValueError("noncontiguous supervised completion")
        suffix = labels[eos[-1]+1:]
        if suffix and (tokenizer is None or tokenizer.decode(suffix).strip()):
            raise ValueError("non-whitespace serialized suffix after terminal EOS")
        if suffix_ids is not None:
            suffix_ids.update(suffix)
    return len(dataset)
