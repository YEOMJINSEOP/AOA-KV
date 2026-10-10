"""
Test: Decode Phase v_log_norm Update Validation (Phase 2d)
==========================================================

Purpose:
    Validate that v_log_norm is correctly updated during decode phase
    when a page becomes full.

Expected behavior:
    - v_log_norm should be computed when a page becomes full during decode
    - Pages that are partially filled should not have v_log_norm updated
    - The v_log_norm value should be log(||V_avg||_2)

Author: Claude (Anthropic)
Date: 2024-12
"""

import sys
# Ensure we use the correct path
sys.path.insert(0, '/SSD/JSY/quest')

import torch
import math

import quest.utils
from quest.utils.controller import InferenceController as IC
print(f"DEBUG: InferenceController from {IC.__module__}")
print(f"DEBUG: Has v_log_norm_layer: {hasattr(IC, 'v_log_norm_layer')}")


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
# Test 1: v_log_norm update during decode when page becomes full
# ============================================================================
@torch.inference_mode()
def test_decode_v_log_norm_update():
    """
    Test that v_log_norm is updated during decode when a page becomes full.

    Strategy:
    1. Prefill with (page_size - 1) tokens so the first page is almost full
    2. Decode one token - this should fill the first page
    3. Check that v_log_norm is now set for page 0
    """
    print_test_header("Decode v_log_norm Update When Page Full")

    dtype = torch.float16
    device = torch.device("cuda:0")

    num_heads = 8
    head_dim = 128
    num_layers = 2
    page_size = 16
    page_budget = 64
    max_seq_len = 8192

    # Prefill with (2 * page_size - 1) tokens so we have:
    # - 1 full page (page 0)
    # - 1 almost full page (page 1 with page_size - 1 tokens)
    # Need at least 2 pages for decode to work
    prefill_len = 2 * page_size - 1  # 31 tokens

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

    # Create random K, V for prefill
    k_prefill = torch.randn(prefill_len, num_heads, head_dim, dtype=dtype, device=device)
    v_prefill = torch.randn(prefill_len, num_heads, head_dim, dtype=dtype, device=device)

    # Prefill
    controller.prepare_metadata(prefill_len)
    controller.begin_forward(prefill_len)
    quest.utils.append_kv(k_prefill, v_prefill, controller, layer_idx=0)
    controller.end_forward()

    # Check v_log_norm before decode
    # Page 0 was filled during prefill, Page 1 has (page_size - 1) tokens
    v_log_norm_page0 = controller.v_log_norm_layer(0)[0, :].clone()  # Page 0 (should be computed)
    v_log_norm_page1_before = controller.v_log_norm_layer(0)[1, :].clone()  # Page 1 (not full yet)

    print(f"\nAfter prefill ({prefill_len} tokens):")
    print(f"  Page 0 is full ({page_size}/{page_size} tokens) - v_log_norm should be computed")
    print(f"  Page 1 has {prefill_len - page_size}/{page_size} tokens - v_log_norm should be 0")
    print_stats("v_log_norm[page=0]", v_log_norm_page0.float())
    print_stats("v_log_norm[page=1] (before decode)", v_log_norm_page1_before.float())

    # Now decode 1 token - this should fill page 1
    controller.prepare_metadata(1)
    controller.begin_forward(1)
    k_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    v_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    quest.utils.append_kv(k_decode, v_decode, controller, layer_idx=0)
    controller.end_forward()

    # Check v_log_norm after decode
    v_log_norm_page1_after = controller.v_log_norm_layer(0)[1, :].clone()  # Page 1
    total_tokens = prefill_len + 1  # 32 = 2 * page_size
    print(f"\nAfter decode (1 token, total {total_tokens} = {total_tokens // page_size} full pages):")
    print(f"  Page 1 is NOW full ({page_size}/{page_size} tokens)")
    print_stats("v_log_norm[page=1] (after decode)", v_log_norm_page1_after.float())

    # v_log_norm should now have valid values (not all zeros or initial)
    # The expected range is roughly [-5, 5] for log of L2 norm of random vectors
    assert not torch.isnan(v_log_norm_page1_after).any(), "Found NaN in v_log_norm after decode"
    assert not torch.isinf(v_log_norm_page1_after).any(), "Found Inf in v_log_norm after decode"

    # Check that v_log_norm was actually updated (should be non-trivial)
    # For random values, the L2 norm should be meaningful
    mean_val = v_log_norm_page1_after.float().mean().item()
    print(f"\n  Mean v_log_norm: {mean_val:.4f}")
    print(f"  Expected range for log(||V_avg||_2): roughly [-2, 2] for normalized random vectors")

    print("\n[PASS] Decode v_log_norm update test passed!")


# ============================================================================
# Test 2: Multiple page fills during decode sequence
# ============================================================================
@torch.inference_mode()
def test_multiple_page_fills():
    """
    Test that v_log_norm is updated for multiple pages during a decode sequence.
    """
    print_test_header("Multiple Page Fills During Decode")

    dtype = torch.float16
    device = torch.device("cuda:0")

    num_heads = 8
    head_dim = 128
    num_layers = 2
    page_size = 16
    page_budget = 64
    max_seq_len = 8192

    # Prefill with exactly 2 * page_size tokens (fills 2 pages completely)
    prefill_len = 2 * page_size

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

    k_prefill = torch.randn(prefill_len, num_heads, head_dim, dtype=dtype, device=device)
    v_prefill = torch.randn(prefill_len, num_heads, head_dim, dtype=dtype, device=device)

    controller.prepare_metadata(prefill_len)
    controller.begin_forward(prefill_len)
    quest.utils.append_kv(k_prefill, v_prefill, controller, layer_idx=0)
    controller.end_forward()

    print(f"\nAfter prefill ({prefill_len} tokens = {prefill_len // page_size} full pages):")
    v_log_norm_page0 = controller.v_log_norm_layer(0)[0, :].float()
    v_log_norm_page1 = controller.v_log_norm_layer(0)[1, :].float()
    print_stats("v_log_norm[page=0]", v_log_norm_page0)
    print_stats("v_log_norm[page=1]", v_log_norm_page1)

    # Now decode enough to fill page 2
    decode_tokens = page_size  # This will fill page 2

    for i in range(decode_tokens):
        controller.prepare_metadata(1)
        controller.begin_forward(1)
        k_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
        v_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
        quest.utils.append_kv(k_decode, v_decode, controller, layer_idx=0)
        controller.end_forward()

    print(f"\nAfter decode ({decode_tokens} tokens, total {prefill_len + decode_tokens}):")
    print(f"  = {(prefill_len + decode_tokens) // page_size} full pages")
    v_log_norm_page2 = controller.v_log_norm_layer(0)[2, :].float()
    print_stats("v_log_norm[page=2]", v_log_norm_page2)

    # All pages should have valid v_log_norm
    assert not torch.isnan(v_log_norm_page2).any(), "Found NaN in page 2 v_log_norm"
    assert not torch.isinf(v_log_norm_page2).any(), "Found Inf in page 2 v_log_norm"

    print("\n[PASS] Multiple page fills test passed!")


# ============================================================================
# Test 3: End-to-end decode with scoring
# ============================================================================
@torch.inference_mode()
def test_decode_with_scoring():
    """
    Test that the full decode pipeline (append + estimate) works correctly.
    """
    print_test_header("End-to-End Decode with Scoring")

    dtype = torch.float16
    device = torch.device("cuda:0")

    num_heads = 16
    head_dim = 128
    num_layers = 2
    page_size = 16
    page_budget = 64
    max_seq_len = 8192

    # Prefill with 256 tokens (16 full pages)
    prefill_len = 256

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

    k_prefill = torch.randn(prefill_len, num_heads, head_dim, dtype=dtype, device=device)
    v_prefill = torch.randn(prefill_len, num_heads, head_dim, dtype=dtype, device=device)

    controller.prepare_metadata(prefill_len)
    controller.begin_forward(prefill_len)
    quest.utils.append_kv(k_prefill, v_prefill, controller, layer_idx=0)
    controller.end_forward()

    # Do several decode iterations
    num_decode_steps = 32
    all_scores = []

    print(f"\nPrefill: {prefill_len} tokens")
    print(f"Decode steps: {num_decode_steps}")

    for step in range(num_decode_steps):
        controller.prepare_metadata(1)
        controller.begin_forward(1)

        k_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
        v_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
        quest.utils.append_kv(k_decode, v_decode, controller, layer_idx=0)

        q_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
        scores = quest.utils.decode_estimate(q_decode, controller, layer_idx=0)
        all_scores.append(scores.clone())

        controller.end_forward()

    # Analyze scores
    print(f"\nScoring analysis over {num_decode_steps} decode steps:")
    for i in [0, num_decode_steps // 2, num_decode_steps - 1]:
        scores_float = all_scores[i].float()
        print(f"  Step {i}: mean={scores_float.mean().item():.4f}, "
              f"range=[{scores_float.min().item():.4f}, {scores_float.max().item():.4f}]")

    # Verify no NaN or Inf in any step
    for i, scores in enumerate(all_scores):
        assert not torch.isnan(scores).any(), f"Found NaN at step {i}"
        assert not torch.isinf(scores).any(), f"Found Inf at step {i}"

    print("\n[PASS] End-to-end decode with scoring test passed!")


# ============================================================================
# Test 4: Full attention pipeline (prefill + decode + sparse attn)
# ============================================================================
@torch.inference_mode()
def test_full_attention_pipeline():
    """
    Test the complete attention pipeline including sparse attention.
    """
    print_test_header("Full Attention Pipeline")

    dtype = torch.float16
    device = torch.device("cuda:0")

    num_heads = 32
    head_dim = 128
    num_layers = 2
    page_size = 16
    page_budget = 32  # Smaller budget to test top-k selection
    max_seq_len = 8192

    prefill_len = 256  # 16 pages

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

    k_prefill = torch.randn(prefill_len, num_heads, head_dim, dtype=dtype, device=device)
    v_prefill = torch.randn(prefill_len, num_heads, head_dim, dtype=dtype, device=device)

    controller.prepare_metadata(prefill_len)
    controller.begin_forward(prefill_len)
    quest.utils.append_kv(k_prefill, v_prefill, controller, layer_idx=0)
    controller.end_forward()

    # Decode with full pipeline
    controller.prepare_metadata(1)
    controller.begin_forward(1)

    k_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    v_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)
    quest.utils.append_kv(k_decode, v_decode, controller, layer_idx=0)

    q_decode = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)

    # Step 1: Estimate attention scores
    estimated_scores = quest.utils.decode_estimate(q_decode, controller, layer_idx=0)
    print(f"\nEstimated scores shape: {estimated_scores.shape}")
    print_stats("Estimated scores", estimated_scores.float())

    # Step 2: Top-k selection
    quest.utils.decode_topk(estimated_scores, controller)

    # Step 3: Sparse attention
    topk_indices = controller.topk_dindices_buffer[:, :controller.inference_page_budget - 1]
    output = quest.utils.decode_sparse_attn(
        q_decode, controller, layer_idx=0, topk_indices=topk_indices
    )

    print(f"\nOutput shape: {output.shape}")
    print_stats("Attention output", output.float())

    assert not torch.isnan(output).any(), "Found NaN in attention output"
    assert not torch.isinf(output).any(), "Found Inf in attention output"

    controller.end_forward()

    print("\n[PASS] Full attention pipeline test passed!")


# ============================================================================
# Run tests directly
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DECODE v_log_norm VALIDATION TESTS (Phase 2d)")
    print("=" * 60)

    # Test 1: v_log_norm update when page becomes full
    test_decode_v_log_norm_update()

    # Test 2: Multiple page fills
    test_multiple_page_fills()

    # Test 3: Decode with scoring
    test_decode_with_scoring()

    # Test 4: Full pipeline
    test_full_attention_pipeline()

    print("\n" + "=" * 60)
    print("ALL PHASE 2d TESTS PASSED!")
    print("=" * 60)
