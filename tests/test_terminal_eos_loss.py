import pytest
import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from ila.terminal_eos_loss import terminal_eos_extra


def sample():
    torch.manual_seed(123)
    logits = torch.randn(2, 6, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 6, 3, 6], [-100, 4, 5, 6, -100, -100]])
    return logits, labels


def loss(logits, labels, weight, denom=None):
    native = ForCausalLMLoss(logits, labels, vocab_size=7, num_items_in_batch=denom)
    return native if weight == 1 else native+(weight-1)*terminal_eos_extra(logits, labels, 6, denom)


def test_weight_one_is_native_loss_and_gradient():
    logits, labels = sample()
    actual = loss(logits, labels, 1)
    expected = ForCausalLMLoss(logits, labels, vocab_size=7)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])


@pytest.mark.parametrize("weight", [1, 10])
def test_accumulation_denominator_matches_concatenated_loss_and_grad(weight):
    logits, labels = sample()
    denom = labels[:, 1:].ne(-100).sum()
    full = loss(logits, labels, weight, denom)
    split = sum(loss(logits[i:i+1], labels[i:i+1], weight, denom) for i in range(2))
    assert torch.allclose(full, split, atol=1e-6)
    assert torch.allclose(torch.autograd.grad(full, logits)[0], torch.autograd.grad(split, logits)[0], atol=1e-7)


def test_only_last_supervised_eos_is_weighted():
    logits, labels = sample()
    extra = terminal_eos_extra(logits, labels, 6)
    grad = torch.autograd.grad(extra, logits)[0]
    assert grad[0, 4].abs().sum() > 0 and grad[1, 2].abs().sum() > 0
    assert grad[0, 2].abs().sum() == 0  # internal supervised EOS is untouched
    assert grad[:, 0].abs().sum() == 0


def test_missing_terminal_eos_rejected():
    logits, labels = sample()
    labels[0, 5] = 4
    with pytest.raises(ValueError):
        terminal_eos_extra(logits, labels, 6)


def test_serialized_whitespace_suffix_does_not_move_weighted_eos():
    logits, labels = sample()
    labels[1] = torch.tensor([-100, 4, 6, 2, -100, -100])
    with pytest.raises(ValueError): terminal_eos_extra(logits,labels,6)
    extra=terminal_eos_extra(logits,labels,6,allowed_suffix_ids=(2,))
    grad=torch.autograd.grad(extra,logits)[0]
    assert grad[1,1].abs().sum()>0
    assert grad[1,2].abs().sum()==0


@pytest.mark.parametrize("weight", [1, 10])
def test_actual_trainer_accumulation_and_gradients(weight):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from train_terminal_eos import trainer_class
    from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments
    class Dataset(torch.utils.data.Dataset):
        n_truncated = 0
        items = [{"input_ids": [1, 2, 3, 6], "labels": [-100, 2, 3, 6]},
                 {"input_ids": [1, 4, 6], "labels": [-100, 4, 6]}]
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]
    model = GPT2LMHeadModel(GPT2Config(vocab_size=7, n_positions=8, n_embd=8, n_layer=1,
        n_head=1, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, eos_token_id=6))
    trainer = trainer_class(weight, 6, {})(model=model, train_dataset=Dataset(),
        args=TrainingArguments(output_dir="unused_test_output", use_cpu=True, report_to=[], gradient_accumulation_steps=2))
    ids = torch.tensor([[1, 2, 3, 6], [1, 4, 6, 0]])
    labels = torch.tensor([[-100, 2, 3, 6], [-100, 4, 6, -100]])
    masks = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    denom = labels[:, 1:].ne(-100).sum()
    assert trainer.model_accepts_loss_kwargs
    inputs = {"input_ids": ids, "labels": labels, "attention_mask": masks}
    full = trainer.compute_loss(model, dict(inputs), num_items_in_batch=denom)
    split = sum(trainer.compute_loss(model, {k:v[i:i+1] for k,v in inputs.items()}, num_items_in_batch=denom) for i in range(2))
    assert torch.allclose(full, split, atol=1e-6)
    params = [p for p in model.parameters() if p.requires_grad]
    full_grad = torch.autograd.grad(full, params)
    split_grad = torch.autograd.grad(split, params)
    assert all(torch.allclose(a,b,atol=2e-6) for a,b in zip(full_grad,split_grad))
