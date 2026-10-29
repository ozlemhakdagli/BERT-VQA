"""H1 VQA network, batches of (image, question, target), and the AdamW + soft-score BCE training loop with early stopping."""
from __future__ import annotations

import copy
from collections.abc import Callable, Iterable

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch import nn


class H1(nn.Module):
    """The VQA network every run trains: question -> 2 stacked BiLSTMs; image -> FC; fusion; MLP classifier (raw logits).

    The question is either word ids [B, L], looked up in a frozen `embedding_matrix`, or precomputed float sequences
    [B, L, input_dim] (e.g. frozen BERT states). As in the original model, the LSTMs run over the whole padded sequence
    unmasked. `fusion`: "product" (classic: image x question, pooled image [B, D]) or "attention" (the question
    attends over image grid tokens [B, T, D] with `glimpses` heads, then product fusion).
    `norm=True` layer-normalizes the image and question vectors right before fusing (keeps the fusions trainable at
    comparable scales; off for the classic head).
    """

    def __init__(self, image_dim: int, num_answers: int, *, fusion: str = "product", embedding_matrix: torch.Tensor | None = None,
                 input_dim: int | None = None, vector_dim: int | None = None, hidden: int = 512, fc: int = 1024, dropout: float = 0.5,
                 glimpses: int = 2, norm: bool = False):
        """Initialize the H1 VQA network architecture.

        Args:
            image_dim: Feature dimension of the input visual representations.
            num_answers: Vocabulary size (number of output answer classes).
            fusion: Fusion type, either 'product' (classic) or 'attention'.
            embedding_matrix: Optional frozen word vector matrix for static word IDs [V, D].
            input_dim: Feature dimension for sequential token inputs (e.g. BERT hidden states).
            vector_dim: Feature dimension for single sentence vector representations.
            hidden: Hidden dimension for each BiLSTM direction. Defaults to 512.
            fc: Dimension of intermediate linear and fusion layers. Defaults to 1024.
            dropout: Dropout probability. Defaults to 0.5.
            glimpses: Number of attention glimpses for attention fusion. Defaults to 2.
            norm: Whether to apply layer normalization to visual and question vectors before fusion.
        """
        super().__init__()
        self.fusion, self.norm, self.embedding, self.vector_dim = fusion, norm, None, vector_dim
        if vector_dim is not None:
            self.question_fc = nn.Linear(vector_dim, fc)
        else:
            if embedding_matrix is not None:
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
                input_dim = embedding_matrix.shape[1]
            self.lstm1 = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
            self.question_fc = nn.Linear(2 * hidden, fc)
        self.image_fc = nn.Linear(image_dim, fc)
        if fusion == "attention":
            self.attend_q, self.attend = nn.Linear(fc, fc), nn.Linear(fc, glimpses)
            self.merge = nn.Linear(glimpses * fc, fc)
        self.head = nn.Sequential(nn.Linear(fc, fc), nn.Tanh(), nn.Dropout(dropout),
                                  nn.Linear(fc, fc), nn.Tanh(), nn.Dropout(dropout), nn.Linear(fc, num_answers))
        self.drop = nn.Dropout(dropout)

    def _question(self, question: torch.Tensor) -> torch.Tensor:
        """Encode question inputs into an intermediate question representation vector.

        Args:
            question: Question inputs (sentence vectors [B, D], word IDs [B, L], or token states [B, L, D]).

        Returns:
            torch.Tensor: Question representation tensor of shape [B, fc].
        """
        if self.vector_dim is not None:
            return self.drop(torch.tanh(self.question_fc(self.drop(question))))
        x = question if self.embedding is None else self.embedding(question)
        x, _ = self.lstm1(x)
        _, (h, _) = self.lstm2(self.drop(x))                                  # final state of each direction
        return self.drop(torch.tanh(self.question_fc(self.drop(torch.cat([h[0], h[1]], dim=1)))))

    def _weights(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute attention softmax weights over spatial image tokens.

        Args:
            v: Transformed image grid token features [B, T, fc].
            q: Encoded question representation vector [B, fc].

        Returns:
            torch.Tensor: Softmax attention distribution over tokens [B, T, glimpses].
        """
        return torch.softmax(self.attend(v * torch.tanh(self.attend_q(q)).unsqueeze(1)), dim=1)   # [B, T, glimpses]

    def attention_weights(self, image: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention distribution over image tokens.

        Args:
            image: Input image grid tokens [B, T, image_dim].
            question: Input question tokens or IDs.

        Returns:
            torch.Tensor: Softmax attention weights over image tokens [B, T, glimpses].
        """
        return self._weights(torch.tanh(self.image_fc(image)), self._question(question))

    def forward(self, image: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the H1 network to compute unnormalized answer logits.

        Args:
            image: Pooled image vectors [B, image_dim] or image token grids [B, T, image_dim].
            question: Question inputs (sentence vectors [B, vector_dim], word IDs [B, L], or token states [B, L, input_dim]).

        Returns:
            torch.Tensor: Unnormalized class logit scores of shape [B, num_answers].
        """
        q, v = self._question(question), torch.tanh(self.image_fc(image))
        if self.fusion == "attention":
            v = self.merge(torch.einsum("btg,btf->bgf", self._weights(v, q), v).flatten(1))
            v = v if self.norm else torch.tanh(v)
        if self.norm:
            v, q = F.layer_norm(v, v.shape[-1:]), F.layer_norm(q, q.shape[-1:])
        return self.head(v * q)


class Rows:
    """Rows of questions indexed by position."""

    def __init__(self, data, device=None):
        """Initialize question row container.

        Args:
            data: Underlying tensor or array containing question inputs.
            device: Target PyTorch device for tensor indexing.
        """
        self.data = data
        self.device = torch.device(device) if device is not None else (
            data.device if isinstance(data, torch.Tensor) else torch.device("cpu")
        )

    def __len__(self) -> int:
        """Return total number of question rows."""
        return len(self.data)

    def take(self, idx: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Extract a batch slice of question rows and move to target device.

        Args:
            idx: Indices of rows to select.

        Returns:
            torch.Tensor: Selected question batch tensor.
        """
        if isinstance(self.data, torch.Tensor):
            out = self.data[torch.as_tensor(idx, device=self.data.device)]
        else:
            out = torch.as_tensor(self.data[idx.cpu().numpy() if isinstance(idx, torch.Tensor) else idx])
        out = out.to(self.device)
        return out.float() if out.is_floating_point() else out


class Pooled:
    """Pre-extracted pooled image vectors."""

    def __init__(self, vectors: torch.Tensor):
        """Initialize pooled image vector wrapper.

        Args:
            vectors: Float tensor of pooled image features [N, D].
        """
        self.vectors = vectors

    def take(self, rows: np.ndarray) -> torch.Tensor:
        """Retrieve pooled image vectors corresponding to the given row indices.

        Args:
            rows: Image row indices to select.

        Returns:
            torch.Tensor: Batch of pooled image vectors.
        """
        return self.vectors[torch.tensor(rows, device=self.vectors.device)]   # a copy: rows may be a read-only pandas array


class Grid:
    """Standardized grid token features."""

    def __init__(self, tokens, mu: torch.Tensor, sd: torch.Tensor, device="cuda"):
        """Initialize standardized grid token feature provider.

        Args:
            tokens: Memory-mapped or tensor grid tokens [N, T, D].
            mu: Per-feature mean tensor for standardization.
            sd: Per-feature standard deviation tensor for standardization.
            device: Target device for batch transfer. Defaults to 'cuda'.
        """
        self.tokens, self.device = tokens, torch.device(device)
        self.mu, self.sd = mu.to(device=self.device, dtype=torch.float32), sd.to(device=self.device, dtype=torch.float32)

    def take(self, rows: np.ndarray) -> torch.Tensor:
        """Extract and standardize grid token features for the specified image rows.

        Args:
            rows: Image row indices to extract.

        Returns:
            torch.Tensor: Standardized grid tokens tensor for the batch.
        """
        unique, inverse = np.unique(rows, return_inverse=True)   # an image's grid is read and standardized once per batch
        out = (torch.as_tensor(self.tokens[unique], device=self.device).float() - self.mu) / self.sd
        out = out.half() if self.device.type == "cuda" else out
        return out[torch.as_tensor(inverse, device=self.device)]


def feature_stats(array, rows: np.ndarray, sample: int = 4096, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-feature mean and standard deviation from a sample of image rows.

    Args:
        array: Array or memory-map of image feature vectors or grids.
        rows: Array of fit image indices from which to sample.
        sample: Maximum number of rows to sample for computing statistics. Defaults to 4096.
        seed: Random seed for sampling. Defaults to 0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of (mean, std) as float32 CPU tensors.
    """
    rows = np.asarray(rows)
    sampled = np.sort(np.random.default_rng(seed).choice(rows, size=min(sample, len(rows)), replace=False))
    scaler = StandardScaler().fit(np.asarray(array[sampled], dtype=np.float32).reshape(-1, array.shape[-1]))
    return torch.as_tensor(scaler.mean_, dtype=torch.float32), torch.as_tensor(scaler.var_, dtype=torch.float32).sqrt().clamp(min=1e-3)


def batches(images, image_rows, questions, targets=None, *, size: int = 512, rng: np.random.Generator | None = None, device="cuda"):
    """Yield batches of (image, question[, target]) for evaluation or training.

    When `rng` is provided, whole images are grouped and shuffled to ensure each image
    is read only once per batch.

    Args:
        images: Image feature container (Pooled or Grid).
        image_rows: Array mapping each question row to its image row index.
        questions: Rows container for questions.
        targets: Optional soft target tensor for training.
        size: Target batch size in number of question items. Defaults to 512.
        rng: Optional numpy Generator for shuffling batches during training.
        device: Target compute device. Defaults to 'cuda'.

    Yields:
        tuple: (image, question) during inference, or (image, question, target) during training.
    """
    image_rows = np.asarray(image_rows)
    if (n := len(image_rows)) == 0:
        return
    if rng is None:
        batch_indices = [np.arange(start, min(start + size, n)) for start in range(0, n, size)]
    else:
        order = np.argsort(image_rows, kind="stable")
        sorted_rows = image_rows[order]
        groups = np.split(order, np.flatnonzero(sorted_rows[1:] != sorted_rows[:-1]) + 1)
        batch_indices, cur, cur_len = [], [], 0
        for g in (groups[i] for i in rng.permutation(len(groups))):
            cur.append(g)
            cur_len += len(g)
            if cur_len >= size:
                batch_indices.append(np.concatenate(cur))
                cur, cur_len = [], 0
        if cur:
            batch_indices.append(np.concatenate(cur))

    for idx in batch_indices:
        image = images.take(image_rows[idx]).to(device)
        question = questions.take(idx)
        if targets is None:
            yield image, question
        else:
            yield image, question, targets[torch.as_tensor(idx, device=targets.device)].float().to(device)


def train(model: nn.Module, epoch_batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
          holdout_score: Callable[[nn.Module], float], *, lr: float, epochs: int, patience: int,
          device: str | torch.device = "cuda", warmup: int = 0) -> tuple[float, int]:
    """Train H1 model using AdamW and soft-score BCE loss with early stopping on holdout accuracy.

    Args:
        model: PyTorch H1 neural network to train.
        epoch_batches: Callable returning an iterator over training batches.
        holdout_score: Callable evaluating model performance on the holdout validation set.
        lr: Initial peak learning rate.
        epochs: Maximum number of training epochs.
        patience: Number of epochs without holdout improvement before stopping.
        device: Training compute device ('cuda' or 'cpu'). Defaults to 'cuda'.
        warmup: Number of initial optimization steps for linear learning rate warmup. Defaults to 0.

    Returns:
        tuple[float, int]: Best holdout accuracy score achieved and total epochs run.
    """
    model.to(device)
    is_cuda = torch.device(device).type == "cuda"
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0)
    best_score, best_state, stale, epochs_run = float("-inf"), None, 0, 0

    for epochs_run in range(1, epochs + 1):
        model.train()
        for image, question, target in epoch_batches():
            optimizer.zero_grad()
            image, question, target = image.to(device), question.to(device), target.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=is_cuda):
                logits = model(image, question)
            loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="sum") / len(target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()

        score = holdout_score(model)
        if best_state is None or score > best_score:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score, epochs_run


def predict(model: nn.Module, batch_iter: Iterable[tuple], device: str | torch.device = "cuda") -> np.ndarray:
    """Predict argmax class indices across all batches.

    Args:
        model: Trained H1 model in evaluation mode.
        batch_iter: Iterator yielding (image, question) batch tuples.
        device: Compute device for inference. Defaults to 'cuda'.

    Returns:
        np.ndarray: 1-D array of integer predicted class indices.
    """
    model.to(device).eval()
    is_cuda = torch.device(device).type == "cuda"
    predictions: list[torch.Tensor] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=is_cuda):
        for batch in batch_iter:
            logits = model(batch[0].to(device), batch[1].to(device))
            predictions.append(torch.argmax(logits, dim=-1))
    return torch.cat(predictions).cpu().numpy() if predictions else np.empty(0, dtype=np.int64)
