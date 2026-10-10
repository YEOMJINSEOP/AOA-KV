"""
Test: v_log_norm Prefill Kernel Validation
==========================================

Purpose:
    Validate that AppendPagedKVCachePrefillKernel correctly computes and stores
    log(||V_avg||_2) for each (page, head) pair.

Expected behavior:
    - For each full page, v_log_norm[page_idx, head_idx] should equal
      log(||mean(V_entries)||_2)
    - Values should be in a reasonable range (not NaN, not Inf)

Author: Claude (Anthropic)
Date: 2024-12
"""

import pytest
from itertools import product

import torch
import torch.nn as nn
import math

import quest.utils


def compute_expected_v_log_norm_cpu(
    v: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """
    CPU reference implementation for v_log_norm calculation.

    Args:
        v: Value tensor, shape [seq_len, num_heads, head_dim]
        page_size: Number of entries per page

    Returns:
        v_log_norm: shape [num_full_pages, num_heads]
        Each entry is log(||V_avg||_2) where V_avg is the channel-wise mean
        of V entries within that page.
    """
    seq_len, num_heads, head_dim = v.shape
    num_full_pages = seq_len // page_size  # Only full pages

    if num_full_pages == 0:
        return torch.empty((0, num_heads), dtype=torch.float32, device=v.device)

    # Reshape to [num_full_pages, page_size, num_heads, head_dim]
    # Only consider full pages
    v_full = v[:num_full_pages * page_size].view(num_full_pages, page_size, num_heads, head_dim)

    # Compute V_avg per page: mean over page_size dimension
    # Shape: [num_full_pages, num_heads, head_dim]
    v_avg = v_full.float().mean(dim=1)

    # Compute L2 norm over head_dim dimension
    # Shape: [num_full_pages, num_heads]
    l2_norm = torch.norm(v_avg, p=2, dim=-1)

    # Apply log with epsilon for numerical stability
    epsilon = 1e-6
    log_norm = torch.log(l2_norm + epsilon)

    return log_norm


def print_test_header(test_name: str):
    """Print formatted test header."""
    print("\n" + "=" * 60)
    print(f"TEST: {test_name}")
    print("=" * 60)


def print_stats(name: str, tensor: torch.Tensor):
    """Print tensor statistics."""
    if tensor.numel() == 0:
        print(f"{name}: (empty)")
        return
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
    print(f"  Mean:  {tensor.mean().item():.6f}")
    print(f"  Std:   {tensor.std().item():.6f}")
    print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
    print(f"  Inf count: {torch.isinf(tensor).sum().item()}")


# ============================================================================
# Test 1: Basic functionality - check values are stored and reasonable
# ============================================================================
parameters_basic = list(product(
    ["float16"],
    [64, 128, 256, 512, 1024],  # seq_len
    [16, 32],  # page_size
))

@pytest.mark.parametrize("dtype_str, seq_len, page_size", parameters_basic)
@torch.inference_mode()
def test_v_log_norm_basic_storage(dtype_str, seq_len, page_size):
    """
    Test that v_log_norm values are stored after prefill and are in valid range.
    """
    print_test_header(f"Basic Storage (seq_len={seq_len}, page_size={page_size})")

    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    num_layers = 2  # Only need 2 layers for testing
    page_budget = 1024
    max_seq_len = 8192

    if seq_len < page_size:
        pytest.skip("seq_len should be >= page_size for at least one full page")

    # Create random V tensor
    k = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)

    # Initialize controller
    controller = quest.utils.InferenceController(
        num_layers,
        num_heads,
        head_dim,
        page_size,
        page_budget,
        max_seq_len,
        dtype,
        device,
    )

    # Run prefill
    controller.prepare_metadata(seq_len)
    controller.begin_forward(seq_len)
    quest.utils.append_kv(k, v, controller, layer_idx=0)
    controller.end_forward()

    # Get v_log_norm buffer for layer 0
    v_log_norm_cuda = controller.v_log_norm_layer(0)

    # Calculate expected number of full pages
    num_full_pages = seq_len // page_size

    print(f"\nResults:")
    print(f"  seq_len: {seq_len}")
    print(f"  page_size: {page_size}")
    print(f"  num_full_pages: {num_full_pages}")
    print(f"  v_log_norm buffer shape: {v_log_norm_cuda.shape}")

    # Check only the pages that should have been filled
    # v_log_norm shape: [max_pages, num_heads]
    v_log_norm_filled = v_log_norm_cuda[:num_full_pages, :]

    print_stats("v_log_norm (filled pages)", v_log_norm_filled.float())

    # Assertions
    assert not torch.isnan(v_log_norm_filled).any(), "Found NaN values in v_log_norm"
    assert not torch.isinf(v_log_norm_filled).any(), "Found Inf values in v_log_norm"

    # Values should be in a reasonable range for log(||V_avg||_2)
    # For fp16 random values ~N(0,1), V_avg should have ||V_avg||_2 ~ sqrt(head_dim) * scale
    # log of that should be roughly in [-10, 10] range
    assert v_log_norm_filled.min() > -20, f"v_log_norm too small: {v_log_norm_filled.min()}"
    assert v_log_norm_filled.max() < 20, f"v_log_norm too large: {v_log_norm_filled.max()}"

    print("\n[PASS] Basic storage test passed!")


# ============================================================================
# Test 2: Numerical correctness - compare with CPU reference
# ============================================================================
parameters_correctness = list(product(
    ["float16"],
    [128, 256, 512],  # seq_len
    [16],  # page_size (fixed for detailed comparison)
))

@pytest.mark.parametrize("dtype_str, seq_len, page_size", parameters_correctness)
@torch.inference_mode()
def test_v_log_norm_correctness(dtype_str, seq_len, page_size):
    """
    Test that v_log_norm values match CPU reference calculation.
    """
    print_test_header(f"Correctness (seq_len={seq_len}, page_size={page_size})")

    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    num_layers = 2
    page_budget = 1024
    max_seq_len = 8192

    # Create random V tensor
    k = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)

    # Initialize controller
    controller = quest.utils.InferenceController(
        num_layers,
        num_heads,
        head_dim,
        page_size,
        page_budget,
        max_seq_len,
        dtype,
        device,
    )

    # Run prefill
    controller.prepare_metadata(seq_len)
    controller.begin_forward(seq_len)
    quest.utils.append_kv(k, v, controller, layer_idx=0)
    controller.end_forward()

    # Get CUDA result
    v_log_norm_cuda = controller.v_log_norm_layer(0)
    num_full_pages = seq_len // page_size
    cuda_result = v_log_norm_cuda[:num_full_pages, :].float()

    # Compute CPU reference
    cpu_result = compute_expected_v_log_norm_cpu(v, page_size)

    print(f"\nComparison:")
    print_stats("CUDA result", cuda_result)
    print_stats("CPU reference", cpu_result)

    # Compare
    diff = (cuda_result - cpu_result).abs()
    print_stats("Absolute difference", diff)

    # Allow some tolerance due to fp16 precision and order of operations
    rtol = 0.1  # 10% relative tolerance
    atol = 0.1  # 0.1 absolute tolerance

    # Check if values are close
    close_mask = torch.isclose(cuda_result, cpu_result, rtol=rtol, atol=atol)
    match_ratio = close_mask.float().mean().item()

    print(f"\n  Match ratio (rtol={rtol}, atol={atol}): {match_ratio * 100:.1f}%")
    print(f"  Max absolute diff: {diff.max().item():.6f}")
    print(f"  Mean absolute diff: {diff.mean().item():.6f}")

    # Show some sample comparisons
    print(f"\n  Sample comparisons (first 5 pages, first 4 heads):")
    for p in range(min(5, num_full_pages)):
        cuda_vals = cuda_result[p, :4].tolist()
        cpu_vals = cpu_result[p, :4].tolist()
        print(f"    Page {p}: CUDA={[f'{x:.3f}' for x in cuda_vals]}")
        print(f"            CPU ={[f'{x:.3f}' for x in cpu_vals]}")

    # Assert reasonable match
    assert match_ratio > 0.8, f"Only {match_ratio*100:.1f}% values match within tolerance"

    print("\n[PASS] Correctness test passed!")


# ============================================================================
# Test 3: Multi-layer consistency
# ============================================================================
@torch.inference_mode()
def test_v_log_norm_multi_layer():
    """
    Test that v_log_norm is computed correctly for multiple layers.
    """
    print_test_header("Multi-layer Consistency")

    dtype = torch.float16
    device = torch.device("cuda:0")

    seq_len = 256
    num_heads = 32
    head_dim = 128
    num_layers = 4
    page_size = 16
    page_budget = 1024
    max_seq_len = 8192

    # Initialize controller
    controller = quest.utils.InferenceController(
        num_layers,
        num_heads,
        head_dim,
        page_size,
        page_budget,
        max_seq_len,
        dtype,
        device,
    )

    # Prepare metadata
    controller.prepare_metadata(seq_len)
    controller.begin_forward(seq_len)

    # Append KV for each layer with different random values
    for layer_idx in range(num_layers):
        k = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)
        quest.utils.append_kv(k, v, controller, layer_idx=layer_idx)

    controller.end_forward()

    num_full_pages = seq_len // page_size

    print(f"\nMulti-layer results:")
    for layer_idx in range(num_layers):
        v_log_norm = controller.v_log_norm_layer(layer_idx)
        filled = v_log_norm[:num_full_pages, :].float()
        print(f"  Layer {layer_idx}: mean={filled.mean().item():.4f}, "
              f"std={filled.std().item():.4f}, "
              f"range=[{filled.min().item():.4f}, {filled.max().item():.4f}]")

        # Basic validity checks
        assert not torch.isnan(filled).any(), f"NaN in layer {layer_idx}"
        assert not torch.isinf(filled).any(), f"Inf in layer {layer_idx}"

    # Check that different layers have different values (they should, since V is random)
    layer0 = controller.v_log_norm_layer(0)[:num_full_pages, :]
    layer1 = controller.v_log_norm_layer(1)[:num_full_pages, :]

    assert not torch.allclose(layer0, layer1), "Layer 0 and 1 should have different v_log_norm values"

    print("\n[PASS] Multi-layer consistency test passed!")


# ============================================================================
# Run tests directly (without pytest)
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("V_LOG_NORM PREFILL VALIDATION TESTS")
    print("=" * 60)

    # Test 1: Basic storage
    test_v_log_norm_basic_storage("float16", 256, 16)

    # Test 2: Correctness
    test_v_log_norm_correctness("float16", 256, 16)

    # Test 3: Multi-layer
    test_v_log_norm_multi_layer()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
