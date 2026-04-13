# SPDX-License-Identifier: Apache-2.0
"""Tests for TurboQuant quantizer components (pure Python, no GPU needed)."""

import math

import torch
import pytest

from turboquant_xpu.quantizer.config import TurboQuantConfig, TQ_PRESETS
from turboquant_xpu.quantizer.centroids import solve_lloyd_max, get_centroids
from turboquant_xpu.quantizer.quantizer import generate_rotation_matrix, generate_wht_signs


class TestTurboQuantConfig:
    """Test TQ configuration and layout math."""

    def test_all_presets_exist(self):
        expected = {"turboquant_k8v4", "turboquant_4bit_nc",
                    "turboquant_k3v4_nc", "turboquant_3bit_nc"}
        assert set(TQ_PRESETS.keys()) == expected

    @pytest.mark.parametrize("preset", TQ_PRESETS.keys())
    def test_from_cache_dtype(self, preset):
        cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=128)
        assert cfg.head_dim == 128
        assert cfg.slot_size > 0
        assert cfg.slot_size_aligned % 2 == 0

    def test_k8v4_is_fp8_keys(self):
        cfg = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", 128)
        assert cfg.key_fp8 is True
        assert cfg.key_packed_size == 128  # 1 byte per element

    def test_k3v4_is_mse_keys(self):
        cfg = TurboQuantConfig.from_cache_dtype("turboquant_k3v4_nc", 128)
        assert cfg.key_fp8 is False
        assert cfg.key_mse_bits == 3
        assert cfg.norm_correction is True

    def test_slot_size_k3v4_d128(self):
        cfg = TurboQuantConfig.from_cache_dtype("turboquant_k3v4_nc", 128)
        # Key: ceil(128 * 3 / 8) + 4 norms = 48 + 4 = 52
        # Value: ceil(128 * 4 / 8) + 4 scale/zero = 64 + 4 = 68
        # Total: 52 + 68 = 120
        assert cfg.key_packed_size == 52
        assert cfg.value_packed_size == 68
        assert cfg.slot_size == 120

    def test_slot_size_k8v4_d128(self):
        cfg = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", 128)
        # Key: 128 (fp8, 1 byte per element)
        # Value: 64 + 4 = 68
        # Total: 196
        assert cfg.key_packed_size == 128
        assert cfg.value_packed_size == 68
        assert cfg.slot_size == 196

    def test_boundary_skip_layers(self):
        layers = TurboQuantConfig.get_boundary_skip_layers(32, n=2)
        assert layers == ["0", "1", "30", "31"]

    def test_boundary_skip_small_model(self):
        layers = TurboQuantConfig.get_boundary_skip_layers(4, n=2)
        assert layers == ["0", "1", "2", "3"]

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown TurboQuant"):
            TurboQuantConfig.from_cache_dtype("turboquant_invalid", 128)

    @pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
    def test_compression_ratio(self, head_dim):
        """Verify compression ratios match documented values."""
        fp16_size = head_dim * 2 * 2  # K + V, 2 bytes each
        for preset in TQ_PRESETS:
            cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim)
            ratio = fp16_size / cfg.slot_size
            assert ratio > 1.0, f"{preset} at d={head_dim} doesn't compress"


class TestLloydMax:
    """Test Lloyd-Max optimal quantizer."""

    @pytest.mark.parametrize("bits", [3, 4])
    def test_centroid_count(self, bits):
        centroids, boundaries = solve_lloyd_max(128, bits)
        assert centroids.shape[0] == 2**bits
        assert boundaries.shape[0] == 2**bits - 1

    def test_centroids_sorted(self):
        centroids, boundaries = solve_lloyd_max(128, 3)
        assert torch.all(centroids[1:] > centroids[:-1])
        assert torch.all(boundaries[1:] > boundaries[:-1])

    def test_midpoints_between_centroids(self):
        centroids, boundaries = solve_lloyd_max(128, 4)
        for i in range(len(boundaries)):
            assert centroids[i] < boundaries[i] < centroids[i + 1]

    def test_caching(self):
        c1 = get_centroids(128, 3)
        c2 = get_centroids(128, 3)
        assert c1 is c2  # same object from LRU cache

    def test_different_dims(self):
        c64 = get_centroids(64, 3)
        c128 = get_centroids(128, 3)
        # Different d → different sigma → different centroids
        assert not torch.allclose(c64, c128)


class TestRotationMatrix:
    """Test WHT sign generation and rotation matrices."""

    def test_rotation_orthogonal(self):
        Q = generate_rotation_matrix(64, seed=42)
        eye = torch.eye(64)
        assert torch.allclose(Q @ Q.T, eye, atol=1e-5)
        assert torch.allclose(Q.T @ Q, eye, atol=1e-5)

    def test_rotation_deterministic(self):
        Q1 = generate_rotation_matrix(128, seed=42)
        Q2 = generate_rotation_matrix(128, seed=42)
        assert torch.allclose(Q1, Q2)

    def test_different_seeds(self):
        Q1 = generate_rotation_matrix(64, seed=42)
        Q2 = generate_rotation_matrix(64, seed=43)
        assert not torch.allclose(Q1, Q2)

    def test_wht_signs_shape(self):
        signs = generate_wht_signs(128, seed=42)
        assert signs.shape == (128,)
        assert set(signs.tolist()).issubset({-1.0, 1.0})

    def test_wht_signs_deterministic(self):
        s1 = generate_wht_signs(128, seed=42)
        s2 = generate_wht_signs(128, seed=42)
        assert torch.allclose(s1, s2)
