"""Focused checks for the wide binary-classification model."""

import io
import unittest

import torch

from model import WideFireMLP


def test_small_model_forward_backward_has_finite_input_and_parameter_gradients():
    torch.manual_seed(17)
    model = WideFireMLP(
        feature_count=7, width=16, expansion=2, blocks=3, dropout=0.15
    )
    inputs = torch.randn(5, 7, requires_grad=True)

    logits = model(inputs)
    assert logits.shape == (5,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    )
    loss.backward()

    assert torch.isfinite(logits).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_default_architecture_exceeds_two_hundred_million_parameters_without_full_allocation():
    with torch.device("meta"):
        model = WideFireMLP()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count > 200_000_000
    assert model.architecture == {
        "feature_count": 96,
        "width": 4096,
        "expansion": 2,
        "blocks": 3,
        "dropout": 0.2,
    }


def test_state_dict_restores_exact_eval_outputs_and_architecture():
    torch.manual_seed(23)
    original = WideFireMLP(
        feature_count=5, width=12, expansion=3, blocks=2, dropout=0.4
    ).eval()
    inputs = torch.randn(4, 5)
    expected = original(inputs)

    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    restored = WideFireMLP(**original.architecture).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))

    actual = restored(inputs)
    assert restored.architecture == original.architecture
    assert torch.equal(actual, expected)


def test_dropout_is_disabled_in_eval_mode():
    torch.manual_seed(31)
    model = WideFireMLP(
        feature_count=4, width=10, expansion=2, blocks=2, dropout=0.75
    ).eval()
    inputs = torch.randn(6, 4)

    assert torch.equal(model(inputs), model(inputs))


def test_initialize_prior_sets_head_to_prevalence_log_odds():
    model = WideFireMLP(feature_count=3, width=8, expansion=2, blocks=1, dropout=0.0)
    model.initialize_prior(0.2)
    assert torch.isclose(model.head.bias[0], torch.logit(torch.tensor(0.2)))


def test_resume_checkpoint_preserves_safe_optimizer_and_rng_states():
    import numpy as np
    import settings
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    model = WideFireMLP(feature_count=3, width=8, expansion=2, blocks=1, dropout=.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    model(torch.randn(4, 3)).square().mean().backward()
    optimizer.step()
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'torch_rng': torch.get_rng_state(), 'numpy_rng': rng.bit_generator.state}, buffer)
    expected_random = rng.random(5)
    expected_torch = torch.rand(5)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = WideFireMLP(**model.architecture)
    restored.load_state_dict(saved['model'])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    restored_optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'])
    assert np.array_equal(rng.random(5), expected_random)
    assert torch.equal(torch.rand(5), expected_torch)
    assert restored_optimizer.state_dict()['param_groups'] == optimizer.state_dict()['param_groups']


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(function)
                              for name, function in sorted(globals().items())
                              if name.startswith('test_') and callable(function))


if __name__ == '__main__':
    unittest.main()
