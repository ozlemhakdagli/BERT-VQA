"""Tests for decoded-image cache, feature extraction, and CNN backbones."""
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import numpy as np
from PIL import Image
import pytest
import timm
import torch
from torch import nn
import torchvision.models
import transformers

from vqa_lab.config import Paths
from vqa_lab.image import (
    _FeatureMap, _qwen_patchify, _tv_feature_map, build_decoded_cache,
    cnn_spec, decoded_dir, extract_batch, extract_features, fetch_weights, image_path, load_cnn,
    open_decoded, open_features, to_model_input,
)


def make_images(tmp_path: Path, n: int, prefix: str = "val2014") -> list[int]:
    """Create n alternating black/white dummy images for testing."""
    ids = list(range(100, 100 + n))
    for i, img_id in enumerate(ids):
        Image.new("RGB", (16, 16), (255, 255, 255) if i % 2 == 0 else (0, 0, 0)).save(image_path(tmp_path, prefix, img_id))
    return ids


def test_decoded_cache(tmp_path: Path) -> None:
    """Verify decoded image array caching, resume logic, and that a stopped build never passes for a finished cache.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    ids = make_images(tmp_path, 4)
    out_dir = tmp_path / "cache"
    with pytest.raises(FileNotFoundError, match="notebook 00"):   # nothing built yet: the message names the notebook that builds it
        open_decoded(out_dir)
    img0, img2, backup = image_path(tmp_path, "val2014", ids[0]), image_path(tmp_path, "val2014", ids[2]), tmp_path / "backup.jpg"
    img0.rename(backup)
    with pytest.raises((FileNotFoundError, OSError)):   # a fresh build stopped in its first shard leaves zeros behind
        build_decoded_cache(tmp_path, "val2014", ids, out_dir, size=8, num_workers=0, shard_size=2)
    with pytest.raises(ValueError, match="decoded cache is incomplete"):
        open_decoded(out_dir)
    backup.rename(img0)
    img2.rename(backup)
    with pytest.raises((FileNotFoundError, OSError)):   # a missing image stops the build after the finished shard
        build_decoded_cache(tmp_path, "val2014", ids, out_dir, size=8, num_workers=0, shard_size=2)
    assert (out_dir / "rows_done.txt").read_text() == "2"
    backup.rename(img2)
    build_decoded_cache(tmp_path, "val2014", ids, out_dir, size=8, num_workers=0, shard_size=2)   # resumes
    got_ids, pixels = open_decoded(out_dir)
    assert got_ids.tolist() == ids and pixels.shape == (4, 8, 8, 3) and pixels.dtype == np.uint8
    assert (pixels[0] == 255).all() and (pixels[1] == 0).all() and (pixels[2] == 255).all()
    with pytest.raises(ValueError):   # other ids into a finished folder
        build_decoded_cache(tmp_path, "val2014", ids[:2], out_dir, size=8, num_workers=0)
    (out_dir / "rows_done.txt").write_text("2")
    with pytest.raises(ValueError, match="decoded cache is incomplete"):
        open_decoded(out_dir)
    (out_dir / "rows_done.txt").write_text("4")
    img0.rename(backup)
    with pytest.raises((FileNotFoundError, OSError)):   # another size starts again from row 0 and stops in its first shard
        build_decoded_cache(tmp_path, "val2014", ids, out_dir, size=6, num_workers=0, shard_size=2)
    with pytest.raises(ValueError, match="decoded cache is incomplete"):   # the old marker "4" must not vouch for the new zeros
        open_decoded(out_dir)
    backup.rename(img0)
    build_decoded_cache(tmp_path, "val2014", ids, out_dir, size=6, num_workers=0)
    pixels = open_decoded(out_dir)[1]
    assert pixels.shape == (4, 6, 6, 3) and (pixels[0] == 255).all() and (pixels[2] == 255).all()


def test_to_model_input_normalizes_and_resizes() -> None:
    """Verify image tensor normalization, mean/std scaling, and spatial resizing."""
    pixels = torch.full((2, 8, 8, 3), 255, dtype=torch.uint8)
    mean, std = (0.5, 0.25, 0.0), (0.5, 0.5, 1.0)
    out4 = to_model_input(pixels, size=4, mean=mean, std=std, mode="bicubic")
    assert out4.shape == (2, 3, 4, 4) and out4.dtype == torch.float32
    assert [out4[:, i].mean().item() for i in range(3)] == pytest.approx([1.0, 1.5, 1.0], abs=1e-4)
    assert to_model_input(pixels[0], size=8, mean=mean, std=std).shape == (1, 3, 8, 8)
    assert to_model_input(torch.zeros((1, 8, 6, 3), dtype=torch.uint8), size=8, mean=mean, std=std).shape == (1, 3, 8, 8)


class _VaryingNet(nn.Module):
    def __init__(self) -> None:
        """Initialise mock CNN module and call counter."""
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating a deterministic spatial feature pattern.

        Args:
            x: Input image tensor.

        Returns:
            torch.Tensor: Synthetic spatial feature map tensor.
        """
        self.calls += 1
        pat = torch.tensor([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]], device=x.device)
        return pat.unsqueeze(0).expand(x.shape[0], -1, -1, -1).contiguous()


class _FakeCNN(nn.Module):
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return dummy spatial features of shape (B, 64, 4, 4).

        Args:
            x: Input image tensor.

        Returns:
            torch.Tensor: Feature map tensor of ones.
        """
        return torch.ones((x.shape[0], 64, 4, 4))


def test_extract_features(tmp_path: Path) -> None:
    """Verify feature extraction pooling, grid flattening, caching, and cache invalidation.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    ids = make_images(tmp_path, 4)
    cache_dir = tmp_path / "cache"
    build_decoded_cache(tmp_path, "val2014", ids, cache_dir, size=8, num_workers=0)

    net = _VaryingNet()
    out_dir = tmp_path / "out_grid"
    extract_features(net, cache_dir, out_dir, size=8, mean=(0, 0, 0), std=(1, 1, 1), grid=2, batch_size=2, device="cpu")
    assert net.calls == 2

    # pooled is spatial MEAN (distinguishes mean from amax/sum)
    pooled = np.load(out_dir / "pooled.npy")
    assert pooled.shape == (4, 2) and pooled.dtype == np.float16
    assert np.allclose(pooled, np.array([[4.0, 5.0]] * 4), atol=1e-2)

    # features is row-major flattened grid [N, grid*grid, C]
    got_ids, feats = open_features(out_dir)
    assert got_ids.tolist() == ids and feats.shape == (4, 4, 2) and feats.dtype == np.float16
    assert np.allclose(feats[0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], atol=1e-2)

    # Second call reuses existing files without invoking net
    extract_features(net, cache_dir, out_dir, size=8, mean=(0, 0, 0), std=(1, 1, 1), grid=2, batch_size=2, device="cpu")
    assert net.calls == 2

    # Extraction without grid
    out_nogrid = tmp_path / "out_nogrid"
    extract_features(net, cache_dir, out_nogrid, size=8, mean=(0, 0, 0), std=(1, 1, 1), grid=None, batch_size=2, device="cpu")
    assert (out_nogrid / "pooled.npy").exists() and not (out_nogrid / "features.npy").exists()
    assert np.load(out_nogrid / "ids.npy").tolist() == ids

    # Folder without ids.npy triggers re-extraction
    (out_nogrid / "ids.npy").unlink()
    calls_before = net.calls
    extract_features(net, cache_dir, out_nogrid, size=8, mean=(0, 0, 0), std=(1, 1, 1), grid=None, batch_size=2, device="cpu")
    assert net.calls > calls_before and (out_nogrid / "ids.npy").exists()


def test_cnn_spec_and_torchvision_feature_maps() -> None:
    """Verify CNN input specifications and torchvision feature map spatial dimensions."""
    assert cnn_spec("torchvision:alexnet") == {
        "size": 224, "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225), "mode": "bilinear",
    }
    timm_spec = cnn_spec("resnet152.tv_in1k")
    assert timm_spec["size"] == 224 and timm_spec["mean"] == pytest.approx((0.485, 0.456, 0.406), abs=1e-4)
    assert isinstance(timm_spec["mean"], tuple) and isinstance(timm_spec["std"], tuple)

    models = {
        "alexnet": (torchvision.models.alexnet(weights=None).eval(), (1, 256, 6, 6)),
        "squeezenet1_1": (torchvision.models.squeezenet1_1(weights=None).eval(), (1, 512, 13, 13)),
        "shufflenet_v2_x1_0": (torchvision.models.shufflenet_v2_x1_0(weights=None).eval(), (1, 1024, 7, 7)),
        "googlenet": (torchvision.models.googlenet(weights=None, aux_logits=False, init_weights=True).eval(), (1, 1024, 7, 7)),
    }
    x = torch.zeros(1, 3, 224, 224)
    for arch, (model, expected_shape) in models.items():
        assert _tv_feature_map(arch, model)(x).shape == expected_shape

    with pytest.raises(ValueError):
        _tv_feature_map("unknown_arch", nn.Identity())


def test_fetch_weights(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify pretrained weight file resolution, caching, and hub downloading.

    Args:
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for mocking network downloads.
    """
    # 1. Existing file is used without download
    sf = paths.raw_dir / "models/cnn/resnet50.a1_in1k/model.safetensors"
    sf.parent.mkdir(parents=True)
    sf.write_text("weights")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("unexpected download"))
    assert fetch_weights("resnet50.a1_in1k", paths) == sf

    # 2. Local .pth wins
    pth = paths.raw_dir / "models/cnn/legacy_xception.tf_in1k/xception.pth"
    pth.parent.mkdir(parents=True)
    pth.write_text("pth_weights")
    assert fetch_weights("legacy_xception.tf_in1k", paths) == pth

    # 3. Failed download raises FileNotFoundError
    download_calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kw: download_calls.append((repo, kw.get("allow_patterns"))))
    with pytest.raises(FileNotFoundError):
        fetch_weights("missing_model", paths)
    assert download_calls == [("timm/missing_model", ["*.json", "model.safetensors"])]

    # 4. timm model not on HF hub downloads from cfg.url
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("unexpected snapshot_download"))
    monkeypatch.setattr(timm, "get_pretrained_cfg", lambda name: SimpleNamespace(hf_hub_id=None, url="https://x/y.pth"))
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda url, dst: Path(dst).write_text("pth"))
    got = fetch_weights("custom_timm_model", paths)
    assert got == paths.raw_dir / "models/cnn/custom_timm_model/y.pth" and got.exists()


def test_load_cnn(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify loading vision models from timm, torchvision, and Hugging Face repositories.

    Args:
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for mocking vision model factories.
    """
    weights = paths.raw_dir / "models/cnn/resnet50.a1_in1k/model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_text("weights")

    seen = {}
    monkeypatch.setattr(timm, "create_model", lambda name, **kw: seen.update(kw) or _FakeCNN())
    mod_timm = load_cnn("resnet50.a1_in1k", paths, device="cpu")
    assert not mod_timm.training and seen["pretrained_cfg_overlay"] == {"file": str(weights), "custom_load": False}
    assert mod_timm(torch.zeros(2, 3, 8, 8)).shape == (2, 64, 4, 4)

    dummy_alex = torchvision.models.alexnet(weights=None)
    monkeypatch.setattr(torchvision.models, "alexnet", lambda weights: dummy_alex)
    mod_tv = load_cnn("torchvision:alexnet", paths, device="cpu")
    assert not mod_tv.training and mod_tv(torch.zeros(1, 3, 224, 224)).shape == (1, 256, 6, 6)

    with pytest.raises(ValueError):
        load_cnn("torchvision:nonexistent", paths, device="cpu")

    hf_dir = paths.raw_dir / "models/google/siglip2-so400m-patch14-384"
    hf_dir.mkdir(parents=True)
    class DummySiglip(nn.Module):
        def forward(self, x: torch.Tensor) -> SimpleNamespace:
            """Simulate SigLIP forward pass returning mock last hidden states.

            Args:
                x: Input image tensor.

            Returns:
                SimpleNamespace: Object containing dummy last_hidden_state tensor.
            """
            return SimpleNamespace(last_hidden_state=torch.ones(x.shape[0], 729, 64))

    monkeypatch.setattr(transformers.SiglipVisionModel, "from_pretrained", lambda path: DummySiglip())
    mod_hf = load_cnn("hf:google/siglip2-so400m-patch14-384", paths, device="cpu")
    assert not mod_hf.training and mod_hf(torch.zeros(1, 3, 384, 384)).shape == (1, 64, 27, 27)

    with pytest.raises(ValueError):
        load_cnn("hf:nonexistent", paths, device="cpu")


def test_modern_vision_encoders(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify feature map adapters for ViT, CLIP, and Qwen patchification.

    Args:
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for mocking Hugging Face models.
    """
    class FakeViT(nn.Module):
        num_prefix_tokens = 1
        patch_embed = type("PatchEmbed", (), {"grid_size": (2, 3)})()

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            """Simulate ViT forward features with prefix tokens.

            Args:
                x: Input image tensor.

            Returns:
                torch.Tensor: Synthetic token embedding tensor.
            """
            tokens = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
            return tokens.unsqueeze(0).repeat(x.shape[0], 1, 1)

    vit_out = _FeatureMap(FakeViT())(torch.zeros(2, 3, 8, 8))
    assert vit_out.shape == (2, 2, 2, 3)
    assert (vit_out[0, :, 0, 0] == 1.0).all() and (vit_out[0, :, 0, 1] == 2.0).all() and (vit_out[0, :, 0, 2] == 3.0).all()
    assert (vit_out[0, :, 1, 0] == 4.0).all() and (vit_out[0, :, 1, 1] == 5.0).all() and (vit_out[0, :, 1, 2] == 6.0).all()

    class FakeCLIP(nn.Module):
        def forward(self, x: torch.Tensor) -> SimpleNamespace:
            """Simulate CLIP vision forward pass with sequence tokens.

            Args:
                x: Input image tensor.

            Returns:
                SimpleNamespace: Object containing mock last_hidden_state tensor.
            """
            tokens = torch.tensor([[99.0, 99.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
            return SimpleNamespace(last_hidden_state=tokens.unsqueeze(0).repeat(x.shape[0], 1, 1))

    expected_clip = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]).unsqueeze(0).repeat(2, 1, 1, 1)
    (paths.raw_dir / "models/openai/clip-vit-large-patch14-336").mkdir(parents=True)
    monkeypatch.setattr(transformers.CLIPVisionModel, "from_pretrained", lambda path: FakeCLIP())
    clip_out = load_cnn("hf:openai/clip-vit-large-patch14-336", paths, "cpu")(torch.zeros(2, 3, 8, 8))
    assert clip_out.shape == (2, 2, 2, 2) and torch.equal(clip_out, expected_clip)

    assert cnn_spec("hf:google/siglip2-so400m-patch14-384")["size"] == 384
    with pytest.raises(ValueError):
        cnn_spec("hf:unknown/repo")

    torch.manual_seed(42)
    x = torch.randn(2, 3, 32, 32)
    pixel_values, grid_thw = _qwen_patchify(x)
    assert pixel_values.shape == (2 * 4, 1536) and grid_thw == [[1, 2, 2]] * 2

    for k in range(4):
        r, c = 16 * (k // 2), 16 * (k % 2)
        patch = x[0, :, r : r + 16, c : c + 16].numpy()
        assert np.allclose(pixel_values[k].numpy(), np.stack([patch, patch], axis=1).reshape(-1))



def test_extraction_batch_and_decoded_dir(paths: Paths) -> None:
    """Verify extraction batch size heuristic and decoded directory path resolution.

    Args:
        paths: Project paths fixture.
    """
    assert extract_batch(224) == 256
    assert extract_batch(448) < extract_batch(288) < 256
    assert 8 <= extract_batch(600) <= 40
    assert decoded_dir(paths, "val2014") == paths.processed_dir / "features/image/decoded448/val2014"
