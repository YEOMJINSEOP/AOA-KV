import os
import sys
import torch
import math
from torch import nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    repeat_kv,
)

from typing import Optional


__all__ = [
    "get_max_min_filters",
    "repeat_filter",
    "update_filter",
    "update_page_filters",
    "compute_attn_weights_for_selection",
    "compute_residual_value_approximation",
    "reconstruct_key_states_with_svd",
    "get_mean_filters",
    "update_mean_filter",
    "update_mean_page_filters",
    "update_l2_norm_sq_page_filters",
    "compute_page_v_mean",
    "compute_page_v_l2_norm_square",
    # ************************************************************************************************
    "local_heavy_hitter_mask_group_consistent_sum",
    "local_heavy_hitter_mask_value_aware_group_consistent_sum",
    # ************************************************************************************************
    "local_heavy_hitter_mask_avg_v_l2norm_group_consistent_sum",
    # ************************************************************************************************
    "compute_page_v_l2_norm_mean",
    "update_v_l2_norm_mean_page_filters",
    # ************************************************************************************************
    "dump_va_figures",
]


_VA_DUMPED_LAYERS = set()


def dump_va_figures(
    layer_id,
    attn_weights_for_selection,
    value_states_page_mean,
    num_key_value_groups,
    chunk_size,
    dump_dir,
    model_label="Mistral-7B-Instruct-v0.2",
    task_label="vt",
    step_label="Step 1",
    exit_after_layers=None,
    target_head=None,
):
    """Dump per-(layer, query head) two-panel PDFs: page-level qK^T max and value L2-norm.

    attn_weights_for_selection: (b, nh, 1, seq_length) — Quest token-level proxy from
        compute_attn_weights_for_selection.
    value_states_page_mean:     (b, nkvh, num_pages, hd) — page-mean of V at KV-head level.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(dump_dir, exist_ok=True)

    b, nh, q_len, seq_length = attn_weights_for_selection.shape
    _, nkvh, num_pages, _ = value_states_page_mean.shape
    assert b == 1 and q_len == 1, "dump_va_figures expects batch=1, q_len=1 (first decode step)"

    # Pad to chunk_size and amax per chunk (mirrors local_heavy_hitter_mask_*)
    pad_len = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn = attn_weights_for_selection
    if pad_len > 0:
        attn = torch.cat(
            [
                attn,
                torch.full(
                    (b, nh, q_len, pad_len),
                    torch.finfo(attn.dtype).min,
                    device=attn.device,
                    dtype=attn.dtype,
                ),
            ],
            dim=-1,
        )
    chunk_amax = attn.view(b, nh, q_len, -1, chunk_size).amax(dim=-1)  # (1, nh, 1, num_pages)
    chunk_amax_np = chunk_amax[0, :, 0, :num_pages].float().cpu().numpy()  # (nh, num_pages)

    v_norm = value_states_page_mean.norm(p=2, dim=-1)  # (1, nkvh, num_pages)
    v_norm_np = v_norm[0].float().cpu().numpy()  # (nkvh, num_pages)

    x = np.arange(num_pages)
    heads_iter = [target_head] if target_head is not None else range(nh)
    emit_html = bool(os.environ.get("VA_FIGURE_OUTPUT_HTML", ""))
    for h in heads_iter:
        kv_h = h // max(num_key_value_groups, 1)
        title_top = (
            f"[Filter-Max] chunk_attn_weights @ {task_label}: {model_label}, "
            f"Layer {layer_id}, Head {h}, {step_label}"
        )
        title_bot = (
            f"[Value-L2-norm] @ {task_label}: {model_label}, "
            f"Layer {layer_id}, Head {h}, {step_label}"
        )
        fig, axes = plt.subplots(2, 1, figsize=(20, 6))
        axes[0].bar(x, chunk_amax_np[h], color="red")
        axes[0].set_title(title_top)
        axes[0].set_xlabel("Chunk Index")
        axes[0].set_ylabel("Attention Weight")
        axes[1].bar(x, v_norm_np[kv_h], color="green")
        axes[1].set_title(title_bot)
        axes[1].set_xlabel("Chunk Index")
        axes[1].set_ylabel("Value-L2-norm AVG")
        fig.tight_layout()
        fig.savefig(os.path.join(dump_dir, f"L{layer_id:02d}_H{h:02d}.pdf"))
        plt.close(fig)

        if emit_html:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            fig_html = make_subplots(
                rows=2, cols=1, vertical_spacing=0.18,
                subplot_titles=[title_top, title_bot],
            )
            fig_html.add_trace(
                go.Bar(
                    x=x, y=chunk_amax_np[h], marker_color="red",
                    hovertemplate="Page %{x}<br>Attention Weight: %{y:.3f}<extra></extra>",
                ),
                row=1, col=1,
            )
            fig_html.add_trace(
                go.Bar(
                    x=x, y=v_norm_np[kv_h], marker_color="green",
                    hovertemplate="Page %{x}<br>Value-L2-norm: %{y:.3f}<extra></extra>",
                ),
                row=2, col=1,
            )
            fig_html.update_xaxes(title_text="Chunk Index", row=2, col=1)
            fig_html.update_yaxes(title_text="Attention Weight", row=1, col=1)
            fig_html.update_yaxes(title_text="Value-L2-norm AVG", row=2, col=1)
            fig_html.update_layout(width=1800, height=650, showlegend=False, bargap=0.1)
            fig_html.write_html(
                os.path.join(dump_dir, f"L{layer_id:02d}_H{h:02d}.html"),
                include_plotlyjs="cdn",
            )

    _VA_DUMPED_LAYERS.add(layer_id)
    n_written = 1 if target_head is not None else nh
    print(f"[VA dump] Layer {layer_id}: wrote {n_written} PDFs to {dump_dir} (total layers done: {len(_VA_DUMPED_LAYERS)})")

    if exit_after_layers is not None and len(_VA_DUMPED_LAYERS) >= exit_after_layers:
        print(f"[VA dump] Reached {exit_after_layers} layers — exiting.")
        sys.exit(0)

# ******************************************************************************************************************************************************************************************************
# Group-consistent versions for Quest (similar to ShadowKV approach)
# ******************************************************************************************************************************************************************************************************

def local_heavy_hitter_mask_group_consistent_sum(attn_weights, token_budget, chunk_size, num_key_value_groups, fixed_selection=False):
    """
    Quest selection with group consistency using SUM aggregation.

    Args:
        attn_weights: (b, nh, q_length, keys)
        token_budget: number of tokens to select
        chunk_size: page size
        num_key_value_groups: number of queries per KV head
        fixed_selection: whether to force selection of first and last pages

    Returns:
        mask_bottom: (b, nh, q_length, seq_length)
        topk: token indices
    """
    # attn_weights (b, nh, query, keys)
    b, nh, q_len, seq_length = attn_weights.shape

    # expend attn_weights to be divisible by chunk_size
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )
    # attn_weights (b, nh, q_length, seq_length + padding_length)

    # Reshape to (b, nkvh, nkvg, q_len, seq_length + padding_length)
    nkvh = nh // num_key_value_groups
    attn_weights_reshaped = attn_weights.view(
        b, nkvh, num_key_value_groups, q_len, attn_weights.shape[-1]
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights_reshaped.reshape(
        b,
        nkvh,
        num_key_value_groups,
        q_len,
        attn_weights_reshaped.shape[-1] // chunk_size,
        chunk_size,
    ).amax(dim=-1)
    # chunk_attn_weights (b, nkvh, nkvg, q_length, num_chunks)

    # GQA: SUM aggregation across queries in the same KV group
    if num_key_value_groups > 1:
        chunk_attn_weights = chunk_attn_weights.sum(dim=2, keepdim=True)
        # chunk_attn_weights (b, nkvh, 1, q_length, num_chunks)

    # Opitonal sink page must selected
    if fixed_selection:
        chunk_attn_weights[..., 0] = torch.finfo(chunk_attn_weights.dtype).max
        chunk_attn_weights[..., -1] = torch.finfo(chunk_attn_weights.dtype).max

    _, topk = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )
    # topk (b, nkvh, 1, q_len, k)

    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)
    # (b, nkvh, 1, q_len, k, chunk_size)

    topk = topk.reshape(b, nkvh, 1, q_len, -1)
    # (b, nkvh, 1, q_len, k * chunk_size)

    # Create mask for (b, nkvh, 1, q_len, seq_len + padding)
    mask_bottom = torch.zeros(
        (b, nkvh, 1, q_len, attn_weights.shape[-1]),
        dtype=torch.bool,
        device=attn_weights.device
    )

    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :, :seq_length]
    # (b, nkvh, 1, q_len, seq_length)

    # Squeeze and replicate for all queries in each KV group
    mask_bottom = mask_bottom.squeeze(2)  # (b, nkvh, q_len, seq_length)
    mask_bottom = repeat_kv(mask_bottom, num_key_value_groups)
    # (b, nh, q_len, seq_length)

    topk = topk.squeeze(2)  # (b, nkvh, q_len, k * chunk_size)
    topk = repeat_kv(topk, num_key_value_groups)
    # (b, nh, q_len, k * chunk_size)

    return mask_bottom, topk


def local_heavy_hitter_mask_value_aware_group_consistent_sum(attn_weights, token_budget, chunk_size, value_states_page_mean, num_key_value_groups, head_dim, fixed_selection=False):
    """
    Quest value-aware selection with group consistency using SUM aggregation.

    Args:
        attn_weights: (b, nh, q_length, keys)
        token_budget: number of tokens to select
        chunk_size: page size
        value_states_page_mean: (b, nkvh, num_pages, hd)
        num_key_value_groups: number of queries per KV head
        head_dim: dimension of each head
        fixed_selection: whether to force selection of first and last pages

    Returns:
        mask_bottom: (b, nh, q_length, seq_length)
    """
    # attn_weights (b, nh, query, keys)
    b, nh, q_len, seq_length = attn_weights.shape
    nkvh = value_states_page_mean.shape[1]

    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )
    # attn_weights (b, nh, q_length, seq_length + padding_length)

    # Reshape to (b, nkvh, nkvg, q_len, seq_length + padding_length)
    attn_weights_reshaped = attn_weights.view(
        b, nkvh, num_key_value_groups, q_len, attn_weights.shape[-1]
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights_reshaped.reshape(
        b,
        nkvh,
        num_key_value_groups,
        q_len,
        attn_weights_reshaped.shape[-1] // chunk_size,
        chunk_size,
    ).amax(dim=-1)
    # (b, nkvh, nkvg, q_len, num_chunks)
    # ****************************************************************************************************
    # ✅ Log 사용 VA 버전 - 구현 효율성 강화 버전 (Ver 3.)
    log_v_norm = torch.log(value_states_page_mean.norm(p=2, dim=-1) + 1e-8)
    log_v_norm = log_v_norm.unsqueeze(2).unsqueeze(3)
    # (b, nkvh, 1, 1, num_chunks)
    
    chunk_attn_weights = 2 * (chunk_attn_weights  / math.sqrt(value_states_page_mean.shape[-1]))
    # ****************************************************************************************************
    # ✅ Log 사용 VA 버전 (Ver 2.)
    # value_states_page_mean = value_states_page_mean.unsqueeze(2).unsqueeze(3)
    # (b, nkvh, 1, 1, num_chunks, hd)
    
    # chunk_attn_weights = 2 * (chunk_attn_weights  / math.sqrt(value_states_page_mean.shape[-1])) + 2 * torch.log(value_states_page_mean.norm(p=2, dim=-1) + 1e-8)
    # hd 제거 버전 (Ver 2-2.)
    # chunk_attn_weights = 2 * (chunk_attn_weights) + 2 * torch.log(value_states_page_mean.norm(p=2, dim=-1) + 1e-8) 
    # ****************************************************************************************************
    # ✅ 원본 VA 버전 (Ver 1.)
    # chunk_attn_weights_exp = torch.exp(chunk_attn_weights / math.sqrt(value_states_page_mean.shape[-1]))
    
    # chunk_attn_weights_exp = chunk_attn_weights_exp.unsqueeze(-1)
    # (b, nkvh, nkvg, q_len, num_chunks, 1)

    # value_states_page_mean_expanded = value_states_page_mean.unsqueeze(2).unsqueeze(3)
    # (b, nkvh, 1, 1, num_chunks, hd)

    # chunk_attn_weights = ((chunk_attn_weights_exp * value_states_page_mean_expanded) ** 2).sum(dim=-1)
    # ******************************************************************************************************

    # GQA: SUM aggregation across queries in the same KV group (AFTER value-aware scoring)
    if num_key_value_groups > 1:
        # **************************************************************************************************************
        # ✅ for Ver 1.
        # chunk_attn_weights = chunk_attn_weights.sum(dim=2, keepdim=True) 
        # **************************************************************************************************************
        # ✅ for Ver 2.
        # chunk_attn_weights = torch.logsumexp(chunk_attn_weights, dim=2, keepdim=True) 
        # **************************************************************************************************************
        # ✅ for Ver 3.
         chunk_attn_weights = torch.logsumexp(chunk_attn_weights, dim=2, keepdim=True)
        # **************************************************************************************************************
        # (b, nkvh, 1, q_len, num_chunks)
    
    # ✅ for Ver 3.
    # **************************************************************************************************************
    chunk_attn_weights = chunk_attn_weights + 2 * log_v_norm
    # **************************************************************************************************************

    # Squeeze nkvg dimension
    chunk_attn_weights = chunk_attn_weights.squeeze(2)  # (b, nkvh, q_len, num_chunks)

    k = min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1))

    if fixed_selection:
        chunk_attn_weights[..., 0] = torch.finfo(chunk_attn_weights.dtype).max
        chunk_attn_weights[..., -1] = torch.finfo(chunk_attn_weights.dtype).max

    # topk page indices
    _, topk = chunk_attn_weights.topk(k, dim=-1)
    # topk (b, nkvh, q_len, k(page_budget))

    # topk token indices
    topk = topk.unsqueeze(-1).repeat(1, 1, 1, 1, chunk_size) * chunk_size + torch.arange(chunk_size, device=topk.device)
    # (b, nkvh, q_len, k, chunk_size)

    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    # (b, nkvh, q_len, k * chunk_size)

    # Create mask at nkvh level
    mask_bottom = torch.zeros(
        (b, nkvh, q_len, attn_weights.shape[-1]),
        dtype=torch.bool,
        device=attn_weights.device
    )

    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]
    # (b, nkvh, q_len, seq_length)

    # Replicate mask for all queries in each KV group
    mask_bottom = repeat_kv(mask_bottom, num_key_value_groups)
    # (b, nh, q_len, seq_length)

    return mask_bottom


def local_heavy_hitter_mask_avg_v_l2norm_group_consistent_sum(
    attn_weights, token_budget, chunk_size, v_l2_norm_mean, num_key_value_groups, head_dim, fixed_selection=False
):
    """
    Quest value-aware selection using exp(logit/sqrt(d)) * mean(||v||_2) per page.
    Group consistent with SUM aggregation.

    Args:
        attn_weights: (b, nh, q_length, keys)
        token_budget: number of tokens to select
        chunk_size: page size
        v_l2_norm_mean: (b, nkvh, num_pages) - mean of L2 norm per page
        num_key_value_groups: number of queries per KV head
        head_dim: dimension of each head
        fixed_selection: whether to force selection of first and last pages

    Returns:
        mask_bottom: (b, nh, q_length, seq_length)
    """
    b, nh, q_len, seq_length = attn_weights.shape
    nkvh = v_l2_norm_mean.shape[1]

    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (b, nh, q_len, padding_length),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )
    # (b, nh, q_len, seq_length + padding_length)

    # Reshape to (b, nkvh, nkvg, q_len, seq_length + padding_length)
    attn_weights_reshaped = attn_weights.view(
        b, nkvh, num_key_value_groups, q_len, attn_weights.shape[-1]
    )

    # chunk attn_weights: max logit per page
    chunk_attn_weights = attn_weights_reshaped.reshape(
        b, nkvh, num_key_value_groups, q_len,
        attn_weights_reshaped.shape[-1] // chunk_size,
        chunk_size,
    ).amax(dim=-1)
    # (b, nkvh, nkvg, q_len, num_chunks)

    # score = exp(z_p / sqrt(d)) * mean(||v||_2)_p
    chunk_scores = torch.exp(chunk_attn_weights / math.sqrt(head_dim))
    # (b, nkvh, nkvg, q_len, num_chunks)

    v_l2_norm_expanded = v_l2_norm_mean.unsqueeze(2).unsqueeze(3)
    # (b, nkvh, 1, 1, num_chunks)

    chunk_scores = chunk_scores * v_l2_norm_expanded
    # (b, nkvh, nkvg, q_len, num_chunks)

    # GQA: SUM aggregation across queries in the same KV group
    if num_key_value_groups > 1:
        chunk_scores = chunk_scores.sum(dim=2, keepdim=True)
        # (b, nkvh, 1, q_len, num_chunks)

    # Squeeze nkvg dimension
    chunk_scores = chunk_scores.squeeze(2)
    # (b, nkvh, q_len, num_chunks)

    k = min(max(3, token_budget // chunk_size), chunk_scores.size(-1))

    if fixed_selection:
        chunk_scores[..., 0] = torch.finfo(chunk_scores.dtype).max
        chunk_scores[..., -1] = torch.finfo(chunk_scores.dtype).max

    # topk page indices
    _, topk = chunk_scores.topk(k, dim=-1)
    # (b, nkvh, q_len, k)

    # topk token indices
    topk = topk.unsqueeze(-1).repeat(1, 1, 1, 1, chunk_size) * chunk_size + torch.arange(chunk_size, device=topk.device)
    # (b, nkvh, q_len, k, chunk_size)

    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    # (b, nkvh, q_len, k * chunk_size)

    # Create mask
    mask_bottom = torch.zeros(
        (b, nkvh, q_len, attn_weights.shape[-1]),
        dtype=torch.bool,
        device=attn_weights.device
    )
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]
    # (b, nkvh, q_len, seq_length)

    # Replicate mask for all queries in each KV group
    mask_bottom = repeat_kv(mask_bottom, num_key_value_groups)
    # (b, nh, q_len, seq_length)

    return mask_bottom

# ************************************************************************************************************************************************************************

def repeat_filter(page_filters: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, nh, num_pages, minmax, hd = page_filters.shape
                    
    if n_rep == 1:
        return page_filters
    page_filters = page_filters[:, :, None, :, :, :].expand(b, nh, n_rep, num_pages, minmax, hd)
    
    return page_filters.reshape(b, nh*n_rep, num_pages, minmax, hd)   

def update_filter(last_filter, key_states, page_size):
        """
        last_filter: (b, nh, 2, hd)
        key_states: (b, nh, 1, hd)
        
        Returns:
            last_filter: (b, nh, 2, hd)
        """

        max_filter = last_filter[:, :, 0:1, :]
        min_filter = last_filter[:, :, 1:2, :]
        """
        (b, nh, 1, hd)
        """

        max_filter = torch.cat([
                max_filter, 
                key_states,
            ],  dim=-2
        ).amax(dim=-2, keepdim=True)
        
        min_filter = torch.cat([
                min_filter, 
                key_states,
            ], dim=-2
        ).amin(dim=-2, keepdim=True)

        last_filter = torch.cat([max_filter, min_filter], dim=-2)
        
        return last_filter

def get_max_min_filters(key_states, page_size):
        """
        Returns:
            page_filters: (b, nh, num_pages, 2(max,min), hd)
        """
        seq_length = key_states.shape[-2]
        num_pages = math.ceil(seq_length / page_size)
        padding_length = page_size - ((seq_length - 1) % page_size + 1)

        # Let's get max filters
        key_states = torch.cat(
            [
                key_states,
                torch.ones(
                    key_states.shape[0], # b
                    key_states.shape[1], # nh
                    padding_length,
                    key_states.shape[3], # hd
                    device=key_states.device,
                    dtype=key_states.dtype
                ) * torch.tensor(torch.finfo(key_states.dtype).min, 
                                 device=key_states.device,
                                 dtype=key_states.dtype)
            ], dim=-2 
        )
        """
        (b, nh, num_pages*page_size, hd)
        """

        max_filters = key_states.reshape(
              key_states.shape[0], # b
              key_states.shape[1], # nh
              num_pages, # num_pages
              page_size,
              key_states.shape[3],# hd
        ).amax(dim=-2, keepdim=True)
        """
        (b, nh, num_pages, 1, hd)
        """

        # Let's get min filters
        key_states = key_states[:, :, :seq_length, :]

        key_states = torch.cat(
            [
                key_states,
                torch.ones(
                    key_states.shape[0], # b
                    key_states.shape[1], # nh
                    padding_length,
                    key_states.shape[3], # hd
                    device=key_states.device,
                    dtype=key_states.dtype
                ) * torch.tensor(torch.finfo(key_states.dtype).max,
                                 device=key_states.device,
                                 dtype=key_states.dtype)
            ], dim=-2 
        )
        """
        (b, nh, num_pages*page_size, hd)
        """

        min_filters = key_states.reshape(
            key_states.shape[0], # b
            key_states.shape[1], # nh
            num_pages, # num_pages
            page_size,
            key_states.shape[3] # hd
        ).amin(dim=-2, keepdim=True)
        """
        (b, nh, num_pages, 1, hd)
        """

        page_filters = torch.cat(
            [
                max_filters,
                min_filters
              ], dim=-2
        )
        """
        (b, nh, num_pages, 2, hd)
        """

        return page_filters

def get_mean_filters(key_states, page_size):
    """
    Returns:
        page_filters: (b, nh, num_pages, 1, hd)
    """
    seq_length = key_states.shape[-2]
    num_pages = math.ceil(seq_length / page_size)
    padding_length = page_size - ((seq_length - 1) % page_size + 1)

    key_states_wo_last_page = key_states[:, :, :(num_pages-1)*page_size, :]
    key_states_last_page = key_states[:, :, (num_pages-1)*page_size:, :]

    mean_filters_wo_last_page = key_states_wo_last_page.reshape(
        key_states.shape[0],  # b
        key_states.shape[1],  # nh
        (num_pages-1),
        page_size,
        key_states.shape[3],  # hd
    ).mean(dim=-2, keepdim=True)
    """
    (b, nh, num_pages-1, 1, hd)
    """
    
    mean_filter_last_page = key_states_last_page.mean(dim=-2, keepdim=True).unsqueeze(3)
    
    mean_filters = torch.cat([mean_filters_wo_last_page, mean_filter_last_page], dim=2)

    return mean_filters

def update_mean_filter(prev_mean: torch.Tensor, new_token: torch.Tensor, n_prev: int):
    """
    Args:
        prev_mean: (b, nh, 1, hd) - 이전까지의 평균
        new_token: (b, nh, 1, hd) - 새로 들어온 token의 key
        n_prev:     int - 이전까지 누적된 token 수
    Returns:
        updated_mean: (b, nh, 1, hd)
    """
    updated_mean = (prev_mean * n_prev + new_token) / (n_prev + 1)
    return updated_mean

def update_mean_page_filters(
    page_filters: torch.Tensor,
    cur_key_states: torch.Tensor, # can be cur_value_states
    past_seq_len: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Update mean page filters with new key states.

    Args:
        page_filters: (b, nkvh, num_pages, 1(mean), hd)
        cur_key_states: (b, nkvh, 1, hd)
        past_seq_len: Previous sequence length
        chunk_size: Size of each page

    Returns:
        Updated page_filters: (b, nkvh, num_pages, 1, hd)
    """
    if past_seq_len % chunk_size != 0:
        # Update existing last page filter
        last_page_len = past_seq_len % chunk_size
        last_filter = page_filters[:, :, -1, :, :]  # (b, nkvh, 1, hd)
        last_filter = update_mean_filter(last_filter, cur_key_states, last_page_len)  # (b, nkvh, 1, hd)

        page_filters = torch.cat(
            [page_filters[:, :, :-1, :, :], last_filter.unsqueeze(2)],
            dim=2
        )  # (b, nkvh, num_pages, 1, hd)
    else:
        # Create new page filter
        last_filter = cur_key_states.unsqueeze(2)  # (b, nkvh, 1, 1, hd)

        page_filters = torch.cat([page_filters, last_filter], dim=2)
        # (b, nkvh, num_pages, 1, hd)

    return page_filters

def update_l2_norm_sq_page_filters(
    l2_norm_sq_per_page: torch.Tensor,
    cur_value_states: torch.Tensor,
    past_seq_len: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Update L2 norm squared per page with new value states.

    Args:
        l2_norm_sq_per_page: (b, nkvh, num_pages)
        cur_value_states: (b, nkvh, 1, hd)
        past_seq_len: Previous sequence length
        chunk_size: Size of each page

    Returns:
        Updated l2_norm_sq_per_page: (b, nkvh, num_pages)
    """
    if past_seq_len % chunk_size != 0:
        # Update existing last page filter
        last_page_len = past_seq_len % chunk_size
        last_l2_norm_sq = l2_norm_sq_per_page[:, :, -1:, ]  # (b, nkvh, 1)

        # Compute L2 norm squared of new token and update mean
        new_l2_norm_sq = (cur_value_states ** 2).sum(dim=-1)  # (b, nkvh, 1)
        last_l2_norm_sq = (last_l2_norm_sq * last_page_len + new_l2_norm_sq) / (last_page_len + 1)

        l2_norm_sq_per_page = torch.cat(
            [l2_norm_sq_per_page[:, :, :-1], last_l2_norm_sq],
            dim=2
        )  # (b, nkvh, num_pages)
    else:
        # Create new page filter
        new_l2_norm_sq = (cur_value_states ** 2).sum(dim=-1)  # (b, nkvh, 1)

        l2_norm_sq_per_page = torch.cat([l2_norm_sq_per_page, new_l2_norm_sq], dim=2)
        # (b, nkvh, num_pages)

    return l2_norm_sq_per_page

def update_page_filters(
    page_filters: torch.Tensor,
    cur_key_states: torch.Tensor,
    past_seq_len: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Update page filters with new key states.

    Args:
        page_filters: (b, nh, num_pages, 2(max,min), hd)
        cur_key_states: (b, nh, 1, hd)
        past_seq_len: Previous sequence length
        chunk_size: Size of each page

    Returns:
        Updated page_filters: (b, nh, num_pages, 2, hd)
    """
    if past_seq_len % chunk_size != 0:
        # Update existing last page filter
        last_filter = page_filters[:, :, -1, :, :]  # (b, nh, 2, hd)
        last_filter = update_filter(last_filter, cur_key_states, chunk_size)  # (b, nh, 2, hd)

        page_filters = torch.cat(
            [page_filters[:, :, :-1, :, :], last_filter.unsqueeze(2)],
            dim=2
        )  # (b, nh, num_pages, 2, hd)
    else:
        # Create new page filter
        last_filter = torch.cat(
            [cur_key_states.unsqueeze(2), cur_key_states.unsqueeze(2)],
            dim=-2
        )  # (b, nh, 1, 2, hd)

        page_filters = torch.cat([page_filters, last_filter], dim=2)
        # (b, nh, num_pages+1, 2, hd)

    return page_filters

def compute_page_v_mean(kv_seq_len, chunk_size, value_states):
    """
    Compute mean value states per page.

    Args:
        kv_seq_len: Current KV sequence length
        chunk_size: Size of each page
        value_states: (b, nh, seq_len, hd)

    Returns:
        value_states_page_mean: (b, nh, num_total_pages, hd)
    """
    seq_length = kv_seq_len
    num_full_chunks = seq_length // chunk_size
    value_states_full = value_states[:, :, :num_full_chunks * chunk_size, :]

    value_states_full = value_states_full.reshape(
        value_states_full.shape[0],
        value_states_full.shape[1],
        num_full_chunks,
        chunk_size,
        value_states_full.shape[3]
    )
    # value_states_full (b, nh, num_full_pages, page_size, hd)

    value_states_full_mean = value_states_full.mean(dim=3)
    # value_states_full_mean (b, nh, num_full_pages, hd)

    if (seq_length % chunk_size != 0):
        value_states_left = value_states[:, :, num_full_chunks * chunk_size:, :]
        value_states_left_mean = value_states_left.mean(dim=2, keepdim=True)
        # value_states_left_mean (b, nh, 1, hd)
        value_states_page_mean = torch.cat([value_states_full_mean, value_states_left_mean], dim=2)
    else:
        value_states_page_mean = value_states_full_mean
    # value_states_page_mean (b, nh, num_total_pages, hd)

    return value_states_page_mean

def compute_page_v_l2_norm_square(kv_seq_len, chunk_size, value_states):
    """
    Compute mean of L2 norm squared of value vectors per page.

    Args:
        kv_seq_len: Current KV sequence length
        chunk_size: Size of each page
        value_states: (b, nh, seq_len, hd)

    Returns:
        value_l2_norm_sq_mean: (b, nh, num_total_pages)
    """
    seq_length = kv_seq_len
    num_full_chunks = seq_length // chunk_size
    value_states_full = value_states[:, :, :num_full_chunks * chunk_size, :]

    value_states_full = value_states_full.reshape(
        value_states_full.shape[0],
        value_states_full.shape[1],
        num_full_chunks,
        chunk_size,
        value_states_full.shape[3]
    )
    # value_states_full (b, nh, num_full_pages, page_size, hd)

    # Compute L2 norm squared for each token: sum(v^2, dim=-1)
    value_l2_norm_sq_full = (value_states_full ** 2).sum(dim=-1)
    # value_l2_norm_sq_full (b, nh, num_full_pages, page_size)

    # Average across tokens in each page
    value_l2_norm_sq_full_mean = value_l2_norm_sq_full.mean(dim=-1)
    # value_l2_norm_sq_full_mean (b, nh, num_full_pages)

    if (seq_length % chunk_size != 0):
        value_states_left = value_states[:, :, num_full_chunks * chunk_size:, :]
        # Compute L2 norm squared for remaining tokens
        value_l2_norm_sq_left = (value_states_left ** 2).sum(dim=-1)
        # value_l2_norm_sq_left (b, nh, num_remaining_tokens)
        value_l2_norm_sq_left_mean = value_l2_norm_sq_left.mean(dim=-1, keepdim=True)
        # value_l2_norm_sq_left_mean (b, nh, 1)
        value_l2_norm_sq_mean = torch.cat([value_l2_norm_sq_full_mean, value_l2_norm_sq_left_mean], dim=-1)
    else:
        value_l2_norm_sq_mean = value_l2_norm_sq_full_mean
    # value_l2_norm_sq_mean (b, nh, num_total_pages)

    return value_l2_norm_sq_mean

def compute_page_v_l2_norm_mean(kv_seq_len, chunk_size, value_states):
    """
    Compute mean of L2 norm of value vectors per page: mean(||v_i||_2) for each page.

    Args:
        kv_seq_len: Current KV sequence length
        chunk_size: Size of each page
        value_states: (b, nkvh, seq_len, hd)

    Returns:
        v_l2_norm_mean: (b, nkvh, num_total_pages)
    """
    seq_length = kv_seq_len
    num_full_chunks = seq_length // chunk_size
    value_states_full = value_states[:, :, :num_full_chunks * chunk_size, :]

    value_states_full = value_states_full.reshape(
        value_states_full.shape[0],
        value_states_full.shape[1],
        num_full_chunks,
        chunk_size,
        value_states_full.shape[3]
    )
    # (b, nkvh, num_full_pages, page_size, hd)

    # L2 norm per token, then average across tokens in each page
    v_l2_norm_full = value_states_full.norm(p=2, dim=-1)
    # (b, nkvh, num_full_pages, page_size)
    v_l2_norm_full_mean = v_l2_norm_full.mean(dim=-1)
    # (b, nkvh, num_full_pages)

    if seq_length % chunk_size != 0:
        value_states_left = value_states[:, :, num_full_chunks * chunk_size:, :]
        v_l2_norm_left = value_states_left.norm(p=2, dim=-1)
        # (b, nkvh, num_remaining_tokens)
        v_l2_norm_left_mean = v_l2_norm_left.mean(dim=-1, keepdim=True)
        # (b, nkvh, 1)
        v_l2_norm_mean = torch.cat([v_l2_norm_full_mean, v_l2_norm_left_mean], dim=-1)
    else:
        v_l2_norm_mean = v_l2_norm_full_mean
    # (b, nkvh, num_total_pages)

    return v_l2_norm_mean

def update_v_l2_norm_mean_page_filters(
    v_l2_norm_mean: torch.Tensor,
    cur_value_states: torch.Tensor,
    past_seq_len: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Update page-level mean of L2 norm with a new value token.

    Args:
        v_l2_norm_mean: (b, nkvh, num_pages)
        cur_value_states: (b, nkvh, 1, hd)
        past_seq_len: Previous sequence length
        chunk_size: Size of each page

    Returns:
        Updated v_l2_norm_mean: (b, nkvh, num_pages)
    """
    # L2 norm of the new token
    new_l2_norm = cur_value_states.norm(p=2, dim=-1).squeeze(-1)
    # (b, nkvh)

    if past_seq_len % chunk_size != 0:
        # Update existing last page
        last_page_len = past_seq_len % chunk_size
        last_mean = v_l2_norm_mean[:, :, -1]  # (b, nkvh)
        updated_mean = (last_mean * last_page_len + new_l2_norm) / (last_page_len + 1)

        v_l2_norm_mean = torch.cat(
            [v_l2_norm_mean[:, :, :-1], updated_mean.unsqueeze(-1)],
            dim=-1
        )
    else:
        # New page
        v_l2_norm_mean = torch.cat(
            [v_l2_norm_mean, new_l2_norm.unsqueeze(-1)],
            dim=-1
        )

    return v_l2_norm_mean

def compute_attn_weights_for_selection(
    query_states: torch.Tensor,
    page_filters: torch.Tensor,
    num_key_value_groups: int,
    chunk_size: int,
    seq_length: int,
    num_pages: int,
) -> torch.Tensor:
    """
    Compute attention weights for token selection using page filters.

    Args:
        query_states: (b, nh, 1, hd)
        page_filters: (b, nh, num_pages, 2, hd)
        num_key_value_groups: Number of key-value groups
        chunk_size: Size of each page
        seq_length: Current sequence length
        num_pages: Number of pages

    Returns:
        quantized_weight: (b, nh, 1, seq_length)
    """
    sign = (query_states > 0) + (~(query_states > 0)) * -1  # (b, nh, 1, hd)
    postive_query = query_states * sign

    max_key = repeat_filter(page_filters, num_key_value_groups) * sign.unsqueeze(2)
    # max_key: (b, nh, num_pages, 2, hd)

    max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        num_pages,
        2,
        max_key.shape[4]
    ).amax(dim=-2)  # (b, nh, num_pages, hd)

    max_key = max_key.unsqueeze(-2).repeat(1, 1, 1, chunk_size, 1)

    max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        -1,
        max_key.shape[-1],
    )[:, :, :seq_length, :]  # (b, nh, seq_length, hd)

    quantized_weight = torch.matmul(
        postive_query.float(),
        max_key.transpose(2, 3).float()
    )  # (b, nh, 1, seq_length)

    return quantized_weight

def compute_residual_value_approximation(
    page_scores: torch.Tensor,
    rest_page_indices: torch.Tensor,
    attn_weights: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
    kv_seq_len: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Compute residual value approximation for non-selected pages.

    Args:
        page_scores: (b, nh, 1, num_chunks) - Attention scores per page
        rest_page_indices: (b, nh, 1, num_chunks - k) - Indices of non-selected pages
        attn_weights: (b, nh, 1, seq_len) - Attention weights for topk tokens
        value_states: (b, nkvh, seq_len, hd) - Value states
        num_key_value_groups: Number of key-value groups
        kv_seq_len: Current KV sequence length
        chunk_size: Size of each page

    Returns:
        attn_output: (b, nh, 1, hd) - Combined topk + residual output
    """
    # Get residual attention scores for non-selected pages
    residual_attn_scores_per_page = torch.gather(
        page_scores,
        dim=-1,
        index=rest_page_indices
    )  # (b, nh, 1, num_chunks - k)

    # Compute mean value per page
    page_v_mean = compute_page_v_mean(
        kv_seq_len,
        chunk_size,
        repeat_kv(value_states, num_key_value_groups)
    )  # (b, nh, num_chunks, hd)

    # Get mean values for non-selected pages
    residual_page_v_mean = torch.gather(
        page_v_mean,
        dim=2,
        index=rest_page_indices.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, page_v_mean.size(-1))
    )  # (b, nh, num_chunks - k, hd)

    # Compute residual page attention output
    residual_page_attn_output = (
        residual_attn_scores_per_page.squeeze(2).unsqueeze(-1) * residual_page_v_mean
    )  # (b, nh, num_chunks - k, hd)
    residual_page_attn_output = residual_page_attn_output.sum(dim=2, keepdim=True)  # (b, nh, 1, hd)

    # Compute topk attention output
    topk_attn_output = torch.matmul(
        attn_weights,
        repeat_kv(value_states, num_key_value_groups)
    )  # (b, nh, 1, hd)

    # Combine topk and residual
    attn_output = topk_attn_output + residual_page_attn_output

    return attn_output

def reconstruct_key_states_with_svd(
    pre_rope_key_states: torch.Tensor,
    pre_rope_query_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rank: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Reconstruct key states using truncated SVD and apply rotary position embedding.

    Args:
        pre_rope_key_states: (b, seq_len, num_kv_heads*head_dim) - Key states before RoPE
        pre_rope_query_states: (b, nh, seq_len, hd) - Query states before RoPE
        cos: Cosine values for RoPE
        sin: Sine values for RoPE
        position_ids: Position indices
        rank: SVD rank for truncation
        num_key_value_heads: Number of key-value heads
        head_dim: Dimension of each head

    Returns:
        reconstructed_key_states: (b, nkvh, seq_len, hd) - Reconstructed key states with RoPE
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    
    bsz = pre_rope_key_states.shape[0]

    # Truncated SVD using torch.svd_lowrank (faster than full SVD)
    # Keep float32 throughout to avoid overflow issues with large singular values
    u, s, v = torch.svd_lowrank(pre_rope_key_states.float(), q=rank, niter=2)
    # U: (b, seq_len, rank)
    # S: (b, rank)
    # V: (b, num_kv_heads*head_dim, rank)

    # Efficient reconstruction: combine S and V first, then reshape
    sv = (s.unsqueeze(-1) * v.transpose(-2, -1)).view(
        bsz, rank, num_key_value_heads, head_dim
    ).permute(0, 2, 1, 3)
    # SV: (b, num_kv_heads, rank, head_dim)

    # Efficient matrix multiplication using einsum
    reconstructed_key_states = torch.einsum('bsr,bkrd->bksd', u, sv)
    # reconstructed_key_states: (b, nkvh, seq_len, hd)

    # Apply rotary position embedding (query_states is computed but not returned)
    _, reconstructed_key_states = apply_rotary_pos_emb(
        pre_rope_query_states, reconstructed_key_states, cos, sin, position_ids
    )

    # Convert back to original dtype for compatibility
    reconstructed_key_states = reconstructed_key_states.to(pre_rope_query_states.dtype)

    return reconstructed_key_states
# *****************************************************************************************************************************************************************************************************
