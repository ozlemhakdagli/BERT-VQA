"""Decoded-image cache, feature extraction, and CNN backbones for VQA."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import InterpolationMode, functional as TF

from vqa_lab.config import Paths, atomic_path
from vqa_lab.data import fetch_archive, fetch_hub

CACHE_SIZE = 448

TV_NODES = {
    "alexnet": "features",
    "squeezenet1_1": "features",
    "shufflenet_v2_x1_0": "conv5",
    "googlenet": "inception5b",
}

HF_ENCODERS = {   # transformers vision encoders: input size, normalisation, resize mode (measured from each repository's preprocessing)
    "google/siglip2-so400m-patch14-384": dict(kind="siglip", size=384, mean=(0.5,) * 3, std=(0.5,) * 3, mode="bilinear"),
    "openai/clip-vit-large-patch14-336": dict(kind="clip", size=336, mean=(0.48145466, 0.4578275, 0.40821073),
                                              std=(0.26862954, 0.26130258, 0.27577711), mode="bicubic"),
    "nvidia/C-RADIOv4-SO400M": dict(kind="radio", size=512, mean=(0.0,) * 3, std=(1.0,) * 3, mode="bicubic"),
    "Qwen/Qwen3-VL-Embedding-2B": dict(kind="qwen3vl", size=448, mean=(0.5,) * 3, std=(0.5,) * 3, mode="bicubic"),
}


def image_path(image_dir: Path, prefix: str, image_id: int) -> Path:
    """Return path to a COCO image file given directory, split prefix, and integer ID.

    Args:
        image_dir: Base directory containing COCO images.
        prefix: Split prefix string (e.g. 'train2014', 'val2014').
        image_id: Integer COCO image identifier.

    Returns:
        Path: Full file path formatted as `COCO_{prefix}_{image_id:012d}.jpg`.
    """
    return Path(image_dir) / f"COCO_{prefix}_{image_id:012d}.jpg"


class _DecodedImages(Dataset):
    """PyTorch Dataset for decoding and resizing raw COCO images into tensors."""

    def __init__(self, image_dir: Path, prefix: str, ids: Sequence[int], size: int):
        """Initialize decoded image dataset.

        Args:
            image_dir: Directory containing COCO JPEG images.
            prefix: Split prefix string (e.g. 'train2014', 'val2014').
            ids: Sequence of integer COCO image IDs.
            size: Target square pixel dimension for resizing.
        """
        self.image_dir, self.prefix, self.ids, self.size = Path(image_dir), prefix, list(ids), size

    def __len__(self) -> int:
        """Return total number of images in this dataset."""
        return len(self.ids)

    def __getitem__(self, i: int) -> torch.Tensor:
        """Load, convert to RGB, resize, and return image as a uint8 Tensor [H, W, 3].

        Args:
            i: Index of the image to retrieve.

        Returns:
            torch.Tensor: Decoded image tensor with shape [size, size, 3].
        """
        path = image_path(self.image_dir, self.prefix, self.ids[i])
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB").resize((self.size, self.size), Image.BICUBIC))
        return torch.from_numpy(arr)


def build_decoded_cache(image_dir: Path, prefix: str, image_ids: Sequence[int], out_dir: Path, *,
                        size: int = CACHE_SIZE, num_workers: int = 12, shard_size: int = 4096) -> None:
    """Decode images into uint8 [N, size, size, 3] memmap cache with ids and progress tracking.

    Args:
        image_dir: Path to directory holding raw COCO JPEG files.
        prefix: Split name prefix (e.g. 'train2014', 'val2014').
        image_ids: Sequence of integer image IDs to decode.
        out_dir: Output cache directory for `pixels.npy` and `ids.npy`.
        size: Target image square dimension. Defaults to CACHE_SIZE (448).
        num_workers: Number of DataLoader worker processes. Defaults to 12.
        shard_size: Number of images processed per checkpointed shard. Defaults to 4096.

    Raises:
        ValueError: If `out_dir` already contains cached IDs that mismatch `image_ids`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(image_ids, dtype=np.int64)
    ids_file, pixels_file, progress = out_dir / "ids.npy", out_dir / "pixels.npy", out_dir / "rows_done.txt"

    if ids_file.exists() and not np.array_equal(np.load(ids_file), ids):
        raise ValueError(f"{out_dir} was built for different image ids")
    np.save(ids_file, ids)

    expected_shape = (len(ids), size, size, 3)
    out = np.lib.format.open_memmap(pixels_file, mode="r+") if pixels_file.exists() else None
    if out is None or out.shape != expected_shape:
        out = np.lib.format.open_memmap(pixels_file, mode="w+", dtype=np.uint8, shape=expected_shape)
        start_row = 0
        progress.write_text("0")   # zeros are no finished cache: open_decoded refuses them (and an old marker must not vouch for them)
    else:
        start_row = int(progress.read_text()) if progress.exists() else 0

    for start in range(start_row, len(ids), shard_size):
        shard = _DecodedImages(image_dir, prefix, ids[start : start + shard_size], size)
        loader = DataLoader(shard, batch_size=64, num_workers=num_workers)
        chunk = np.concatenate([batch.numpy() for batch in loader], axis=0)
        out[start : start + len(chunk)] = chunk
        out.flush()
        progress.write_text(str(start + len(chunk)))
    out.flush()


def open_decoded(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Open memory-mapped decoded pixel array and corresponding image IDs.

    Args:
        out_dir: Path to directory holding `ids.npy` and `pixels.npy`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (ids [N], pixels [N, S, S, 3] uint8).

    Raises:
        FileNotFoundError: If `ids.npy` is missing in `out_dir`.
        ValueError: If `rows_done.txt` indicates an incomplete cache build.
    """
    out_dir = Path(out_dir)
    if not (out_dir / "ids.npy").is_file():
        raise FileNotFoundError(f"{out_dir}: no decoded image cache; build it with notebook 00 (00_data)")
    ids = np.load(out_dir / "ids.npy")
    progress = out_dir / "rows_done.txt"
    if progress.exists():
        done = int(progress.read_text())
        if done != len(ids):
            raise ValueError(f"{out_dir}: decoded cache is incomplete ({done} of {len(ids)} images); "
                             "finish it with build_decoded_cache (notebook 00_data)")
    # copy-on-write: torch.from_numpy wants a writable array; nothing is ever written back to the file
    return ids, np.load(out_dir / "pixels.npy", mmap_mode="c")


def to_model_input(pixels: torch.Tensor, size: int, mean: Sequence[float], std: Sequence[float], mode: str = "bicubic") -> torch.Tensor:
    """Convert decoded uint8 pixel tensor [B, H, W, 3] to normalized model input float32 [B, 3, size, size].

    Scales pixel values to [0, 1], resizes using antialiasing, and applies channel normalization.

    Args:
        pixels: uint8 image tensor of shape [H, W, 3] or [B, H, W, 3].
        size: Target square spatial dimension (height and width).
        mean: Sequence of per-channel normalization means.
        std: Sequence of per-channel normalization standard deviations.
        mode: Interpolation mode ('bicubic' or 'bilinear'). Defaults to 'bicubic'.

    Returns:
        torch.Tensor: Float32 normalized tensor with shape [B, 3, size, size].
    """
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    x = pixels.permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0   # a division (to_dtype multiplies by 1/255: other bits)
    x = TF.resize(x, [size, size], interpolation=InterpolationMode(mode), antialias=True)   # a no-op at the same size
    return TF.normalize(x, mean=list(mean), std=list(std))


def _hf_repo(name: str) -> str:
    """Extract and validate the Hugging Face repository from an `hf:<repo>` encoder name.

    Args:
        name: Model identifier prefixed with 'hf:'.

    Returns:
        str: Repository name string.

    Raises:
        ValueError: If repository is not found in HF_ENCODERS.
    """
    repo = name.split(":", 1)[1]
    if repo not in HF_ENCODERS:
        raise ValueError(f"Unknown HuggingFace model: {repo}")
    return repo


def cnn_spec(name: str) -> dict:
    """Retrieve model input preprocessing specification (size, mean, std, mode).

    Supports models from timm, torchvision, and Hugging Face.

    Args:
        name: Model identifier string.

    Returns:
        dict: Preprocessing specification dictionary containing size, mean, std, mode.

    Raises:
        ValueError: If model name cannot be resolved.
    """
    if name.startswith("hf:"):
        repo = _hf_repo(name)
        return {k: v for k, v in HF_ENCODERS[repo].items() if k != "kind"}
    if name.startswith("torchvision:"):
        return {
            "size": 224,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "mode": "bilinear",
        }

    import timm

    cfg = timm.get_pretrained_cfg(name)
    if cfg is None:
        raise ValueError(f"Unknown pretrained timm model: {name}")
    size = (cfg.test_input_size or cfg.input_size)[1]
    return {"size": int(size), "mean": tuple(cfg.mean), "std": tuple(cfg.std),
            "mode": "bicubic" if cfg.interpolation == "bicubic" else "bilinear"}


def fetch_weights(name: str, paths: Paths) -> Path:
    """Obtain local weights file for a timm model, downloading from Hugging Face if missing.

    Args:
        name: timm model name identifier.
        paths: Project paths configuration.

    Returns:
        Path: Path to the local safetensors or pth weights file.

    Raises:
        FileNotFoundError: If weights file cannot be found or downloaded.
    """
    folder = paths.raw_dir / "models/cnn" / name
    weights_file = folder / "model.safetensors"
    if weights_file.exists():
        return weights_file
    if local := sorted(folder.glob("*.pth")):   # weights that are not on the hub (e.g. the original Xception file)
        return local[0]
    import timm

    cfg = timm.get_pretrained_cfg(name)
    if cfg and not getattr(cfg, "hf_hub_id", None) and getattr(cfg, "url", None):
        return fetch_archive(cfg.url, folder / Path(cfg.url).name)
    fetch_hub(f"timm/{name}", folder, needed=["model.safetensors"], patterns=["*.json", "model.safetensors"])
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found at {weights_file}")
    return weights_file


class _FeatureMap(nn.Module):
    """Wraps a timm model so forward(x) returns net.forward_features(x) [B, C, h, w]."""

    def __init__(self, net: nn.Module):
        """Initialize feature map wrapper.

        Args:
            net: Backbone neural network module implementing `forward_features`.
        """
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and reshape spatial feature map from the backbone model.

        Args:
            x: Normalized input image batch [B, 3, H, W].

        Returns:
            torch.Tensor: Feature map tensor with shape [B, C, h, w].
        """
        out = self.net.forward_features(x)
        if out.ndim == 4:
            return out
        prefix = getattr(self.net, "num_prefix_tokens", 0)
        tokens = out[:, prefix:]
        grid = self.net.patch_embed.grid_size
        gh, gw = (grid, grid) if isinstance(grid, int) else grid
        b, _, c = tokens.shape
        return tokens.reshape(b, gh, gw, c).permute(0, 3, 1, 2).contiguous()


class _TokenGrid(nn.Module):
    """Wraps a transformers vision model so forward(x) returns its patch tokens as a square map [B, C, s, s]."""

    def __init__(self, model: nn.Module, field: str, drop: int = 0):
        """Initialize token grid wrapper.

        Args:
            model: Underlying transformers vision encoder module.
            field: Name of output attribute containing sequence tokens (e.g. 'last_hidden_state').
            drop: Number of leading prefix tokens to drop (e.g. 1 for CLS token).
        """
        super().__init__()
        self.model, self.field, self.drop = model, field, drop   # drop: leading class tokens (CLIP has one)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract sequence tokens and reshape them into a square spatial grid [B, C, s, s].

        Args:
            x: Normalized input image batch [B, 3, H, W].

        Returns:
            torch.Tensor: Spatial token grid tensor with shape [B, C, s, s].
        """
        tokens = getattr(self.model(x), self.field)[:, self.drop:]
        b, n, c = tokens.shape
        s = int(n ** 0.5)
        return tokens.reshape(b, s, s, c).permute(0, 3, 1, 2).contiguous()


QWEN_PATCH, QWEN_FRAMES, QWEN_MERGE = 16, 2, 2   # Qwen3-VL: 16-px patches, a still image counts as 2 frames, 2x2 patches -> 1 token


def _qwen_patchify(x: torch.Tensor) -> tuple[torch.Tensor, list[list[int]]]:
    """Patchify image batch [B, 3, H, W] into Qwen2/3-VL pixel_values [B*h*w, 1536] and grid_thw, in the vision tower's own
    patch order (each 2x2 merge block contiguous); 1536 = 3 channels * 2 frames * 16 * 16 pixels.

    Args:
        x: Image batch tensor [B, 3, H, W].

    Returns:
        tuple[torch.Tensor, list[list[int]]]: Tuple of (pixel_values, grid_thw).
    """
    b, c, h_img, w_img = x.shape
    patch, frames, merge = QWEN_PATCH, QWEN_FRAMES, QWEN_MERGE
    h, w = h_img // patch, w_img // patch
    clip = x.unsqueeze(1).expand(-1, frames, -1, -1, -1)
    reshaped = clip.reshape(b, frames, c, h // merge, merge, patch, w // merge, merge, patch)
    permuted = reshaped.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
    pixel_values = permuted.reshape(b * h * w, c * frames * patch * patch)
    grid_thw = [[1, h, w]] * b
    return pixel_values, grid_thw


class _QwenVisionGrid(nn.Module):
    """Wraps Qwen3-VL vision tower to return feature map [B, C, h/2, w/2]."""

    def __init__(self, visual: nn.Module):
        """Initialize Qwen vision tower wrapper.

        Args:
            visual: Vision tower module of Qwen3-VL.
        """
        super().__init__()
        self.visual = visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process patchified images through Qwen vision tower into a feature map [B, C, gh, gw].

        Args:
            x: Input image batch [B, 3, H, W].

        Returns:
            torch.Tensor: Spatial feature map tensor with shape [B, C, gh, gw].
        """
        b = x.shape[0]
        pixel_values, grid_thw = _qwen_patchify(x)
        grid_thw_t = torch.as_tensor(grid_thw, device=x.device, dtype=torch.int64)
        out = self.visual(pixel_values, grid_thw=grid_thw_t)
        tokens = getattr(out, "pooler_output", out)
        gh = (x.shape[2] // QWEN_PATCH) // QWEN_MERGE
        gw = (x.shape[3] // QWEN_PATCH) // QWEN_MERGE
        c = tokens.shape[-1]
        return tokens.reshape(b, gh, gw, c).permute(0, 3, 1, 2).contiguous()


class _NodeOutput(nn.Module):
    """Wraps a torchvision feature extractor so forward(x) returns its "out" node [B, C, h, w]."""

    def __init__(self, net: nn.Module):
        """Initialize torchvision feature extractor wrapper.

        Args:
            net: Torchvision intermediate node feature extractor module.
        """
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the intermediate 'out' feature map from torchvision extractor.

        Args:
            x: Normalized image batch tensor.

        Returns:
            torch.Tensor: Feature map tensor from the designated output node.
        """
        return self.net(x)["out"]


def _tv_feature_map(arch: str, model: nn.Module) -> nn.Module:
    """Construct feature map extractor module for supported torchvision architectures.

    Args:
        arch: Architecture name key in `TV_NODES`.
        model: Torchvision model instance.

    Returns:
        nn.Module: Feature extractor returning the designated intermediate node output.

    Raises:
        ValueError: If `arch` is not present in `TV_NODES`.
    """
    if arch not in TV_NODES:
        raise ValueError(f"Unsupported torchvision architecture: {arch}")
    from torchvision.models.feature_extraction import create_feature_extractor

    return _NodeOutput(create_feature_extractor(model, {TV_NODES[arch]: "out"}))


def load_cnn(name: str, paths: Paths, device: str | torch.device = "cuda") -> nn.Module:
    """Load pretrained CNN or vision encoder in evaluation mode returning spatial feature maps.

    Supports models from Hugging Face ('hf:'), torchvision ('torchvision:'), and timm.

    Args:
        name: Model identifier string.
        paths: Project paths configuration.
        device: Device on which to allocate the model. Defaults to 'cuda'.

    Returns:
        nn.Module: Loaded model in eval mode outputting feature maps [B, C, h, w].

    Raises:
        ValueError: If model prefix or architecture is unknown.
    """
    if name.startswith("hf:"):
        repo = _hf_repo(name)
        kind = HF_ENCODERS[repo]["kind"]
        folder = fetch_hub(repo, paths.raw_dir / "models" / repo,
                           patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja", "pytorch_model.bin"])
        import transformers

        if kind == "siglip":
            net = _TokenGrid(transformers.SiglipVisionModel.from_pretrained(str(folder)), "last_hidden_state")
        elif kind == "clip":
            net = _TokenGrid(transformers.CLIPVisionModel.from_pretrained(str(folder)), "last_hidden_state", drop=1)
        elif kind == "radio":   # NVIDIA's model code from the downloaded folder (trust_remote_code); input in [0, 1]
            net = _TokenGrid(transformers.AutoModel.from_pretrained(str(folder), trust_remote_code=True), "features")
        else:   # qwen3vl: only the vision tower is kept
            net = _QwenVisionGrid(transformers.AutoModel.from_pretrained(str(folder)).visual)
        return net.eval().to(device)

    if name.startswith("torchvision:"):
        arch = name.split(":", 1)[1]
        import torchvision.models

        if not hasattr(torchvision.models, arch):
            raise ValueError(f"Unknown torchvision model: {arch}")
        torch.hub.set_dir(str(paths.raw_dir / "models/torchvision"))
        model = getattr(torchvision.models, arch)(weights="DEFAULT")
        return _tv_feature_map(arch, model).eval().to(device)

    import timm

    weights_file = fetch_weights(name, paths)
    net = timm.create_model(name, pretrained=True, num_classes=0, pretrained_cfg_overlay={"file": str(weights_file), "custom_load": False})
    return _FeatureMap(net).eval().to(device)


def extract_features(net: Callable[[torch.Tensor], torch.Tensor], cache_dir: Path, out_dir: Path, *, size: int,
                     mean: Sequence[float], std: Sequence[float], mode: str = "bicubic", grid: int | None = None,
                     batch_size: int = 256, device: str | torch.device = "cuda") -> None:
    """Extract features from the decoded image cache into `out_dir`.

    Writes `pooled.npy` [N, D] (the feature map's spatial mean), with `grid` also
    `features.npy` [N, grid*grid, D] (the map average-pooled to grid x grid), and `ids.npy`.
    Extraction is skipped when all target files exist.

    Args:
        net: Model or callable that computes feature maps from normalized images.
        cache_dir: Directory containing decoded image cache (`pixels.npy` and `ids.npy`).
        out_dir: Destination directory to store feature files.
        size: Target image square dimension for input.
        mean: Per-channel normalization mean.
        std: Per-channel normalization standard deviation.
        mode: Interpolation mode ('bicubic' or 'bilinear'). Defaults to 'bicubic'.
        grid: Optional spatial grid size for token extraction (e.g. 7 for 7x7).
        batch_size: Batch size for feature extraction inference. Defaults to 256.
        device: Computation device ('cuda' or 'cpu'). Defaults to 'cuda'.
    """
    out_dir = Path(out_dir)
    pooled_file, features_file, ids_file = out_dir / "pooled.npy", out_dir / "features.npy", out_dir / "ids.npy"
    if pooled_file.exists() and ids_file.exists() and (grid is None or features_file.exists()):
        return

    dev = torch.device(device)
    if isinstance(net, nn.Module):
        net = net.to(dev).eval()

    cache_ids, pixels = open_decoded(Path(cache_dir))
    n_images = len(pixels)

    pooled_out, feats_out = None, None   # memmaps created at the first batch, when the feature width is known

    def read_batch(s: int) -> np.ndarray:
        """Read a slice of pixels from the memory-mapped decoded cache.

        Args:
            s: Starting image index.

        Returns:
            np.ndarray: Batch slice array of decoded pixels.
        """
        return np.array(pixels[s : s + batch_size])

    features_ctx = atomic_path(features_file) if grid is not None else nullcontext()
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda")
    with atomic_path(pooled_file) as pooled_tmp, features_ctx as features_tmp:
        with ThreadPoolExecutor(max_workers=1) as ex, torch.inference_mode(), autocast:
            fut = ex.submit(read_batch, 0) if n_images > 0 else None   # the next batch is read from disk while the GPU works
            for start in range(0, n_images, batch_size):
                next_start = start + batch_size
                next_fut = ex.submit(read_batch, next_start) if next_start < n_images else None
                batch_pixels = torch.from_numpy(fut.result()).to(dev)
                fut = next_fut
                feat = net(to_model_input(batch_pixels, size=size, mean=mean, std=std, mode=mode))
                feat_f = feat.float()

                pooled_chunk = feat_f.mean(dim=(2, 3)).half().cpu().numpy()
                if pooled_out is None:
                    pooled_out = np.lib.format.open_memmap(pooled_tmp, mode="w+", dtype=np.float16,
                                                           shape=(n_images, pooled_chunk.shape[1]))
                pooled_out[start : start + len(pooled_chunk)] = pooled_chunk

                if grid is not None:
                    feats_chunk = (torch.nn.functional.adaptive_avg_pool2d(feat_f, grid)
                                   .flatten(2).transpose(1, 2).half().cpu().numpy())
                    if feats_out is None:
                        feats_out = np.lib.format.open_memmap(features_tmp, mode="w+", dtype=np.float16,
                                                              shape=(n_images, grid * grid, feats_chunk.shape[-1]))
                    feats_out[start : start + len(feats_chunk)] = feats_chunk

        if pooled_out is not None:
            pooled_out.flush()
        if feats_out is not None:
            feats_out.flush()
        del pooled_out, feats_out

    with atomic_path(ids_file) as tmp:
        np.save(tmp, cache_ids)


def open_features(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Open image IDs and memory-mapped token features from cache directory.

    Args:
        out_dir: Directory containing `ids.npy` and `features.npy`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (image ids [N], token array [N, T, D]).
    """
    out_dir = Path(out_dir)
    return np.load(out_dir / "ids.npy"), np.load(out_dir / "features.npy", mmap_mode="r")


def decoded_dir(paths: Paths, split: str) -> Path:
    """Return the path to the decoded image cache for a specific COCO split.

    Args:
        paths: Project paths configuration.
        split: Split name (e.g. 'train2014', 'val2014').

    Returns:
        Path: Path to the decoded images directory (`processed_dir / 'features/image/decoded448/<split>'`).
    """
    return paths.image_features_dir / f"decoded{CACHE_SIZE}" / split


def pooled_split(paths: Paths, key: str, split: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load image IDs and pooled feature vectors for an encoder and split.

    Checks both `pooled` and `grid` directories for cached arrays.

    Args:
        paths: Project paths configuration.
        key: Encoder identifier key.
        split: Dataset split ('train2014' or 'val2014').

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Tuple of (ids, pooled_vectors) if found, else None.
    """
    for sub in ("pooled", "grid"):
        d = paths.image_features_dir / sub / key / split
        if (d / "pooled.npy").is_file() and (d / "ids.npy").is_file():
            return np.load(d / "ids.npy"), np.load(d / "pooled.npy")
    return None


def grid_split(paths: Paths, key: str, split: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load image IDs and memory-mapped grid token features for an encoder and split.

    Args:
        paths: Project paths configuration.
        key: Encoder identifier key.
        split: Dataset split ('train2014' or 'val2014').

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Tuple of (ids, grid_tokens) if complete cache exists, else None.
    """
    d = paths.image_features_dir / "grid" / key / split
    return open_features(d) if (d / "features.npy").is_file() and (d / "ids.npy").is_file() else None


def extract_batch(size: int) -> int:
    """Calculate optimal extraction batch size scaled by image resolution.

    Uses 256 for 224px images and scales down inversely with the square of resolution,
    ensuring higher resolution images fit within GPU memory.

    Args:
        size: Image spatial dimension in pixels.

    Returns:
        int: Batch size for feature extraction (minimum 8).
    """
    return max(8, int(256 * (224 / size) ** 2))
