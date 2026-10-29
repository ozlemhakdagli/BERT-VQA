"""Tests for training loops, loss functions, batching, and model architectures."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from vqa_lab.train import Grid, H1, Pooled, Rows, batches, feature_stats, predict, train

DIMS = dict(image_dim=6, num_answers=5, hidden=8, fc=16)


@pytest.mark.parametrize("fusion,is_grid", [("product", False), ("attention", True)])
@pytest.mark.parametrize("q_kind", ["ids", "seq", "vec"])
def test_output_shape_and_eval_determinism(fusion, is_grid, q_kind):
    """Verify H1 output tensor shape, frozen parameters, and deterministic eval behavior.

    Args:
        fusion: Fusion mechanism type string.
        is_grid: Whether the visual input represents a spatial feature grid.
        q_kind: Question representation kind ('ids', 'seq', or 'vec').
    """
    img = torch.randn(4, 9, 6) if is_grid else torch.randn(4, 6)
    kw, q = {
        "ids": (dict(embedding_matrix=torch.randn(20, 8)), torch.randint(0, 20, (4, 7))),
        "seq": (dict(input_dim=12), torch.randn(4, 7, 12)),
        "vec": (dict(vector_dim=5), torch.randn(4, 5)),
    }[q_kind]
    m = H1(**DIMS, fusion=fusion, dropout=0.1, **kw).eval()
    if q_kind == "vec":
        assert not hasattr(m, "lstm1") and not hasattr(m, "lstm2")
    if q_kind == "ids":   # the word embedding is frozen, everything else trains
        assert not m.embedding.weight.requires_grad
        assert all(p.requires_grad for n, p in m.named_parameters() if n != "embedding.weight")
    out = m(img, q)
    assert out.shape == (4, 5)
    assert torch.equal(out, m(img, q)) and torch.allclose(m(img[:1], q[:1]), out[:1], atol=1e-5)
    if fusion == "attention":
        with pytest.raises(RuntimeError):
            m(torch.randn(4, 6), q)


def test_attention_weights_form_distribution_and_depend_on_question():
    """Verify attention weights form a valid probability distribution and vary with question."""
    torch.manual_seed(0)
    model = H1(**DIMS, embedding_matrix=torch.randn(20, 8), fusion="attention", dropout=0.0).eval()
    img, q1, q2 = torch.randn(1, 9, 6), torch.randint(0, 20, (1, 7)), torch.randint(0, 20, (1, 7))
    w1, w2 = model.attention_weights(img, q1), model.attention_weights(img, q2)
    assert w1.shape == (1, 9, 2) and torch.allclose(w1.sum(1), torch.ones(1, 2), atol=1e-5) and not torch.allclose(w1, w2)


class DummyModel(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 2):
        """Initialise dummy linear regression module for training tests.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, image, question):
        """Compute linear projection of image features.

        Args:
            image: Image feature tensor.
            question: Question feature tensor (ignored).

        Returns:
            torch.Tensor: Linear projection output tensor.
        """
        return self.linear(image)


@pytest.mark.parametrize("scores,epochs,expected_epochs,expected_best,best_idx", [
    ([0.1, 0.5, 0.3, 0.2, 0.9], 10, 4, 0.5, 1),
    ([0.1, 0.2, 0.3, 0.4], 3, 3, 0.3, 2),
])
def test_early_stopping_and_rising_score(scores, epochs, expected_epochs, expected_best, best_idx):
    """Verify early stopping triggers on plateau and restores model checkpoint of best epoch.

    Args:
        scores: Sequence of holdout scores per epoch.
        epochs: Maximum number of epochs configured.
        expected_epochs: Expected number of epochs run before stopping.
        expected_best: Expected best score achieved.
        best_idx: Index of epoch corresponding to the best score.
    """
    model, snapshots = DummyModel(), []

    def holdout_score(m):
        """Simulate holdout evaluation and record model state checkpoint.

        Args:
            m: Model being evaluated.

        Returns:
            float: Simulated score for the current epoch.
        """
        snapshots.append(copy.deepcopy(m.state_dict()))
        return scores[len(snapshots) - 1]

    epoch_b = lambda: [(torch.randn(8, 4), torch.randn(8, 4), torch.zeros(8, 2))]
    best_score, epochs_run = train(model, epoch_b, holdout_score, lr=1e-2, epochs=epochs, patience=2, device="cpu")
    assert epochs_run == expected_epochs and best_score == pytest.approx(expected_best)
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k], snapshots[best_idx][k])
    if best_idx != len(snapshots) - 1:
        assert not torch.equal(model.state_dict()["linear.weight"], snapshots[-1]["linear.weight"])


def test_warmup_keeps_first_steps_small():
    """Verify learning rate warmup restricts parameter updates in initial steps."""
    def moved(warmup):
        """Measure aggregate parameter movement after 1 epoch under specified warmup steps.

        Args:
            warmup: Number of warmup steps.

        Returns:
            float: Sum of absolute parameter changes.
        """
        torch.manual_seed(0)
        model = DummyModel(4, 3)
        start = [p.detach().clone() for p in model.parameters()]
        b = lambda: [(torch.randn(8, 4), torch.zeros(8, 1), torch.eye(3)[torch.randint(0, 3, (8,))]) for _ in range(3)]
        train(model, b, lambda m: 0.0, lr=1e-2, epochs=1, patience=1, device="cpu", warmup=warmup)
        return sum((p - s).abs().sum().item() for p, s in zip(model.parameters(), start))

    assert moved(warmup=1000) < 0.05 * moved(warmup=0)


def test_every_epoch_trains_in_train_mode():
    """Verify training loop ensures model is in training mode during every epoch."""
    class ModeModel(DummyModel):
        def __init__(self):
            """Initialise model and training mode log list."""
            super().__init__()
            self.modes: list[bool] = []

        def forward(self, image, question):
            """Record training flag status and perform linear projection.

            Args:
                image: Image feature tensor.
                question: Question feature tensor.

            Returns:
                torch.Tensor: Linear projection output tensor.
            """
            self.modes.append(self.training)
            return self.linear(image)

    model = ModeModel()
    batches_fn = lambda: [(torch.randn(4, 4), torch.randn(4, 4), torch.zeros(4, 2)) for _ in range(2)]
    _, epochs_run = train(model, batches_fn, lambda m: (m.eval(), 0.5)[1], lr=1e-2, epochs=2, patience=2, device="cpu")
    assert epochs_run == 2 and len(model.modes) == 4 and all(model.modes)


def test_predict_returns_argmax_in_batch_order():
    """Verify predict returns argmax classification predictions preserving batch order."""
    class EyeModel(nn.Module):
        def forward(self, image, question):
            """Identity forward pass returning image features directly.

            Args:
                image: Image feature tensor.
                question: Question feature tensor.

            Returns:
                torch.Tensor: Image tensor unchanged.
            """
            return image

    preds = predict(EyeModel(), [
        (torch.tensor([[10.0, 1.0, 0.0], [0.0, 5.0, 2.0]]), torch.zeros(2, 1)),
        (torch.tensor([[1.0, 2.0, 8.0], [0.0, 9.0, 3.0], [7.0, 2.0, 1.0]]), torch.zeros(3, 1)),
    ], device="cpu")
    assert preds.dtype == np.int64 and np.array_equal(preds, np.array([0, 1, 2, 1, 0], dtype=np.int64))
    empty = predict(EyeModel(), [], device="cpu")
    assert empty.dtype == np.int64 and len(empty) == 0


def test_batches_keep_alignment_in_order_or_grouped_by_image():
    """Verify batches maintain image-question alignment and group questions by image."""
    rows = np.array([0, 1, 0, 2, 2, 2, 3, 1, 0, 3])
    images, questions = Pooled(torch.arange(4.0).unsqueeze(1).repeat(1, 3)), Rows(torch.arange(10).unsqueeze(1))
    targets, rng = torch.arange(10.0).unsqueeze(1), np.random.default_rng(0)
    seen = list(batches(images, rows, questions, targets, size=4, device="cpu"))                 # evaluation: in order
    shuffled = list(batches(images, rows, questions, targets, size=4, rng=rng, device="cpu"))   # training: grouped, shuffled
    again = list(batches(images, rows, questions, targets, size=4, rng=rng, device="cpu"))
    assert [len(b[1]) for b in seen] == [4, 4, 2]
    order = [torch.cat([b[1] for b in run]).flatten().tolist() for run in (seen, shuffled, again)]
    assert order[0] == list(range(10)) and sorted(order[1]) == list(range(10)) and len({tuple(o) for o in order}) == 3
    for img, question, tgt in seen + shuffled:
        assert torch.equal(question.float(), tgt)
        assert torch.equal(img[:, 0], torch.as_tensor(rows)[question.flatten()].float())
    batch_of = {int(i): b for b, (_, question, _) in enumerate(shuffled) for i in question.flatten()}
    for r in range(4):
        assert len({batch_of[i] for i in np.flatnonzero(rows == r)}) == 1   # an image's questions share one batch


def test_feature_stats_and_grid_standardisation():
    """Verify feature mean/std computation, variance clamping, and grid standardisation."""
    for shape in ((20, 5), (20, 4, 5)):   # pooled vectors and grids: one statistic per feature dimension
        for t in feature_stats(np.ones(shape, dtype=np.float32), rows=np.arange(20)):
            assert t.shape == (5,) and t.dtype == torch.float32 and t.device.type == "cpu"
    arr = np.full((10, 3), 1e9, dtype=np.float32)
    arr[:5] = 2.0
    mu_rows, _ = feature_stats(arr, rows=np.array([0, 1, 2, 3, 4]))
    assert torch.allclose(mu_rows, torch.full((3,), 2.0))
    _, sd_clamp = feature_stats(np.full((10, 2), 3.0, dtype=np.float32), rows=np.arange(10))
    assert torch.allclose(sd_clamp, torch.full((2,), 1e-3))
    x_tiny = np.array([[100.0 - 1e-6, 1.0], [100.0 + 1e-6, 3.0]], dtype=np.float32)
    mu_tiny, sd_tiny = feature_stats(x_tiny, rows=np.arange(2))
    assert sd_tiny[0].item() == pytest.approx(1e-3) and sd_tiny[1].item() == pytest.approx(1.0)
    assert torch.isfinite(((torch.from_numpy(x_tiny) - mu_tiny) / sd_tiny).half()).all()
    assert torch.allclose(feature_stats(np.ones((1, 2), dtype=np.float32), rows=np.arange(1))[1], torch.full((2,), 1e-3))
    arr_rand = np.random.default_rng(42).normal(size=(50, 4)).astype(np.float32)
    m1, s1 = feature_stats(arr_rand, rows=np.arange(50), sample=10, seed=7)
    m2, s2 = feature_stats(arr_rand, rows=np.arange(50), sample=10, seed=7)
    sampled = np.sort(np.random.default_rng(7).choice(np.arange(50), size=10, replace=False))
    assert torch.allclose(m1, torch.from_numpy(arr_rand[sampled].mean(axis=0)))
    assert torch.allclose(s1, torch.from_numpy(arr_rand[sampled].std(axis=0, ddof=0)))
    assert torch.equal(m1, m2) and torch.equal(s1, s2)

    tokens = np.random.default_rng(0).normal(5.0, 2.0, size=(6, 4, 3)).astype(np.float16)
    out = Grid(tokens, torch.full((3,), 5.0), torch.full((3,), 2.0), device="cpu").take(np.array([3, 1, 3]))   # repeated rows
    assert out.shape == (3, 4, 3) and out.dtype == torch.float32
    assert torch.allclose(out, (torch.as_tensor(tokens[[3, 1, 3]]).float() - 5.0) / 2.0, atol=1e-2)


def test_loss_is_summed_over_answers_and_gradients_are_clipped(monkeypatch):
    """Verify BCE loss is summed across answers and gradients are clipped to max norm.

    Args:
        monkeypatch: Monkeypatch fixture for intercepting clip_grad_norm_.
    """
    class BiasModel(nn.Module):
        def __init__(self, k: int = 16):
            """Initialise bias parameter model.

            Args:
                k: Dimension of the trainable bias parameter.
            """
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(k))

        def forward(self, image, question):
            """Return expanded bias parameters matching batch size.

            Args:
                image: Image tensor used for batch dimension.
                question: Question feature tensor.

            Returns:
                torch.Tensor: Expanded bias tensor.
            """
            return self.bias.expand(len(image), -1)

    k = 16
    model = BiasModel(k)
    recorded_before, recorded_after = [], []
    real_clip = torch.nn.utils.clip_grad_norm_

    def wrapped_clip(parameters, max_norm, *args, **kwargs):
        """Wrapper tracking bias gradient before and after norm clipping.

        Args:
            parameters: Model parameters to clip.
            max_norm: Maximum gradient norm.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Total norm of parameters before clipping.
        """
        params = list(parameters)
        recorded_before.append(model.bias.grad.clone())
        res = real_clip(params, max_norm, *args, **kwargs)
        recorded_after.append(model.bias.grad.clone())
        return res

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", wrapped_clip)
    train(model, lambda: [(torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, k))] * 2, lambda m: 0.0,
          lr=1e-3, epochs=1, patience=1, device="cpu")
    assert len(recorded_before) == 2 and len(recorded_after) == 2
    for before, after in zip(recorded_before, recorded_after):   # the second batch starts from fresh gradients, not on top of the first
        assert torch.allclose(before, torch.full((k,), 0.5), atol=1e-2)   # the optimiser has moved the bias a little
        assert torch.linalg.vector_norm(after).item() == pytest.approx(1.0)
