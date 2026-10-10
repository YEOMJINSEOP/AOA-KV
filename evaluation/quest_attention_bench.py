import math
import numpy as np
import warnings
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.cache_utils import DynamicCache

from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention

############################################################ KIVI ################################################
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))

from filtering.filter_util_bench import *  # import local_heavy_hitter_mask variations

import json

from collections import defaultdict

##################################################################################################################
# Group-consistent version of Quest using SUM aggregation
def forward_quest_group_consistent_sum(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    input_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    mean_l2_diff=None

    bsz, q_len, _ = hidden_states.size()

    query_states = (self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2))
    key_states = (self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))
    value_states = (self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[-1]

    cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is None: # Prefill
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            None,
            q_len,
            dropout=0.0
        )

        if self.layer_id >= 2:
            page_filters = get_max_min_filters(key_states, self.chunk_size)
            past_key_value = (
                key_states, value_states, page_filters, kv_seq_len
            ) if use_cache else None
        else:
            past_key_value = (
                key_states, value_states, kv_seq_len
            ) if use_cache else None

    else: # Decode
        (
            past_key_states, past_value_states,
        ) = past_key_value[0:2]

        cur_key_states = key_states

        key_states = torch.cat([past_key_states, key_states], dim=2)
        value_states = torch.cat([past_value_states, value_states], dim=2)

        attn_weights = torch.matmul(
            query_states,
            repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # ******************************************************** QUEST ****************************************************** (1) update last filter
        if self.layer_id >= 2:
            seq_length = kv_seq_len
            num_pages = math.ceil(seq_length / self.chunk_size)

            page_filters = update_page_filters(
                past_key_value[-2], cur_key_states, past_key_value[-1], self.chunk_size
            )  # (b, nh, num_pages, 2, hd)
        # *********************************************************************************************************************

        # ******************************************************** QUEST ****************************************************** (2) criticality approximation & top-K selection
            attn_weights_for_selection = compute_attn_weights_for_selection(
                query_states, page_filters,
                self.num_key_value_groups, self.chunk_size, seq_length, num_pages
            )  # (b, nh, 1, seq_length)

            token_budget = min(kv_seq_len, self.token_budget)

            if token_budget > 0:
                # GROUP-CONSISTENT SUM VERSION
                mask_bottom, _ = local_heavy_hitter_mask_group_consistent_sum(
                    attn_weights_for_selection, token_budget, self.chunk_size, self.num_key_value_groups, self.fixed
                )
            else:
                mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

            mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
        # *********************************************************************************************************************
            # FOR CHECKING ✅: True Attn Weights
            # *************************************************************************************
            # true_attn_weights = attn_weights
            # true_attn_weights = nn.functional.softmax(true_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # true_attn_output = torch.matmul(
            #     true_attn_weights,
            #     repeat_kv(value_states, self.num_key_value_groups)
            # )  # (b, nh, 1, hd)
            # *************************************************************************************
            attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(
            attn_weights,
            repeat_kv(value_states, self.num_key_value_groups)
        )
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        if self.quest and self.layer_id >= 2:
            past_key_value = (
                key_states, value_states, page_filters, kv_seq_len
            ) if use_cache else None
        else:
            past_key_value = (
                key_states, value_states, kv_seq_len
            ) if use_cache else None

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value, mean_l2_diff

# Group-consistent version of Quest value-aware using SUM aggregation
def forward_quest_value_aware_group_consistent_sum(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    input_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    query_states = (self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2))
    key_states = (self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))
    value_states = (self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[-1]

    cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is None: # Prefill
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            None,
            q_len,
            dropout=0.0
        )

        if self.layer_id >= 2:
            page_filters = get_max_min_filters(key_states, self.chunk_size)
            value_states_page_mean = compute_page_v_mean(kv_seq_len, self.chunk_size, value_states)

            past_key_value = (
                key_states, value_states, value_states_page_mean, page_filters, kv_seq_len
            ) if use_cache else None
        else:
            past_key_value = (
                key_states, value_states, kv_seq_len
            ) if use_cache else None

    else: # Decode
        (
            past_key_states, past_value_states,
        ) = past_key_value[0:2]

        if self.layer_id >= 2:
            cur_key_states = key_states
            cur_value_states = value_states
            value_states_page_mean = past_key_value[-3]
            value_states_page_mean = update_mean_page_filters(
                value_states_page_mean.unsqueeze(-2), cur_value_states, past_key_value[-1], self.chunk_size
            ).squeeze(-2)

        key_states = torch.cat([past_key_states, key_states], dim=2)
        value_states = torch.cat([past_value_states, value_states], dim=2)

        attn_weights = torch.matmul(
            query_states,
            repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # ******************************************************** QUEST ****************************************************** - QUEST decode(1): update last filter
        if self.layer_id >= 2:
            seq_length = kv_seq_len
            num_pages = math.ceil(seq_length / self.chunk_size)

            page_filters = update_page_filters(
                past_key_value[-2], cur_key_states, past_key_value[-1], self.chunk_size
            )  # (b, nh, num_pages, 2, hd)
        # *********************************************************************************************************************

        # ******************************************************** QUEST ****************************************************** - QUEST decode(2): criticality approximation
            attn_weights_for_selection = compute_attn_weights_for_selection(
                query_states, page_filters, self.num_key_value_groups,
                self.chunk_size, seq_length, num_pages
            )  # (b, nh, 1, seq_length)

            # --- Optional dump for the per-page qK^T vs V L2-norm motivation figure ---
            _va_dump_dir = os.environ.get("VA_FIGURE_DUMP_DIR", "")
            if _va_dump_dir:
                if not hasattr(self, "_va_dump_step"):
                    self._va_dump_step = 0
                _target_layer_env = os.environ.get("VA_FIGURE_TARGET_LAYER", "")
                _target_layer = int(_target_layer_env) if _target_layer_env else None
                _target_head_env = os.environ.get("VA_FIGURE_TARGET_HEAD", "")
                _target_head = int(_target_head_env) if _target_head_env else None
                if self._va_dump_step == 0 and (_target_layer is None or self.layer_id == _target_layer):
                    _exit_after = int(os.environ.get("VA_FIGURE_EXIT_AFTER_LAYERS", "30"))
                    dump_va_figures(
                        layer_id=self.layer_id,
                        attn_weights_for_selection=attn_weights_for_selection.detach(),
                        value_states_page_mean=value_states_page_mean.detach(),
                        num_key_value_groups=self.num_key_value_groups,
                        chunk_size=self.chunk_size,
                        dump_dir=_va_dump_dir,
                        exit_after_layers=_exit_after,
                        target_head=_target_head,
                    )
                self._va_dump_step += 1
            # --- end dump hook ---

            token_budget = min(kv_seq_len, self.token_budget)

            if token_budget > 0:
                # GROUP-CONSISTENT SUM VERSION
                mask_bottom = local_heavy_hitter_mask_value_aware_group_consistent_sum(
                    attn_weights_for_selection, token_budget, self.chunk_size, value_states_page_mean, self.num_key_value_groups, self.head_dim, self.fixed
                )
            else:
                mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

            mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
            # FOR CHECKING ✅: True Attn Weights
            # *************************************************************************************
            # true_attn_weights = attn_weights
            # true_attn_weights = nn.functional.softmax(true_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # true_attn_output = torch.matmul(
            #     true_attn_weights,
            #     repeat_kv(value_states, self.num_key_value_groups)
            # )  # (b, nh, 1, hd)
            # *************************************************************************************
            attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)
        # *********************************************************************************************************************

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(
            attn_weights,
            repeat_kv(value_states, self.num_key_value_groups)
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        if self.quest and self.layer_id >= 2:
            past_key_value = (
                key_states, value_states, value_states_page_mean, page_filters, kv_seq_len
            ) if use_cache else None
        else:
            past_key_value = (
                key_states, value_states, kv_seq_len
            ) if use_cache else None

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

# Value-aware version using exp(logit/sqrt(d)) * mean(||v_i||_2) per page
def forward_quest_avg_of_V_l2norm_group_consistent_sum(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    input_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    query_states = (self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2))
    key_states = (self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))
    value_states = (self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[-1]

    cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is None:  # Prefill
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            None,
            q_len,
            dropout=0.0
        )

        if self.layer_id >= 2:
            page_filters = get_max_min_filters(key_states, self.chunk_size)
            v_l2_norm_mean = compute_page_v_l2_norm_mean(kv_seq_len, self.chunk_size, value_states)

            past_key_value = (
                key_states, value_states, v_l2_norm_mean, page_filters, kv_seq_len
            ) if use_cache else None
        else:
            past_key_value = (
                key_states, value_states, kv_seq_len
            ) if use_cache else None

    else:  # Decode
        (
            past_key_states, past_value_states,
        ) = past_key_value[0:2]

        if self.layer_id >= 2:
            cur_key_states = key_states
            cur_value_states = value_states
            v_l2_norm_mean = past_key_value[-3]
            v_l2_norm_mean = update_v_l2_norm_mean_page_filters(
                v_l2_norm_mean, cur_value_states, past_key_value[-1], self.chunk_size
            )

        key_states = torch.cat([past_key_states, key_states], dim=2)
        value_states = torch.cat([past_value_states, value_states], dim=2)

        attn_weights = torch.matmul(
            query_states,
            repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # QUEST decode(1): update last page filter
        if self.layer_id >= 2:
            seq_length = kv_seq_len
            num_pages = math.ceil(seq_length / self.chunk_size)

            page_filters = update_page_filters(
                past_key_value[-2], cur_key_states, past_key_value[-1], self.chunk_size
            )  # (b, nh, num_pages, 2, hd)

        # QUEST decode(2): criticality approximation with avg V L2 norm
            attn_weights_for_selection = compute_attn_weights_for_selection(
                query_states, page_filters, self.num_key_value_groups,
                self.chunk_size, seq_length, num_pages
            )  # (b, nh, 1, seq_length)

            token_budget = min(kv_seq_len, self.token_budget)

            if token_budget > 0:
                mask_bottom = local_heavy_hitter_mask_avg_v_l2norm_group_consistent_sum(
                    attn_weights_for_selection, token_budget, self.chunk_size,
                    v_l2_norm_mean, self.num_key_value_groups, self.head_dim, self.fixed
                )
            else:
                mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

            mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
            attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(
            attn_weights,
            repeat_kv(value_states, self.num_key_value_groups)
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        if self.quest and self.layer_id >= 2:
            past_key_value = (
                key_states, value_states, v_l2_norm_mean, page_filters, kv_seq_len
            ) if use_cache else None
        else:
            past_key_value = (
                key_states, value_states, kv_seq_len
            ) if use_cache else None

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


global layer_id
layer_id = None  # Will be initialized based on model's actual num_hidden_layers

def enable_quest_attention_eval(model, args):
    global layer_id
    # Initialize layer_id based on model's actual layer count (only once)
    if layer_id is None:
        if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            layer_id = model.config.num_hidden_layers
            print(f"🔧 layer_id initialized to {layer_id} (from model.config.num_hidden_layers)")
        else:
            layer_id = 32  # fallback
            print(f"🔧 layer_id initialized to {layer_id} (fallback)")

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_quest_attention_eval(
                module,
                args,
            )

        if isinstance(module, (LlamaAttention, MistralAttention, Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention)):
            # For longchat model
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward

            model._modules[name].quest = args.quest
            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size
            model._modules[name].fixed = args.fixed
            
            # model._modules[name].kivi = args.kivi
            # model._modules[name].k_bits = args.k_bits
            # model._modules[name].v_bits = args.v_bits
            # model._modules[name].group_size = args.group_size
            # model._modules[name].residual_length = args.residual_length
            
            # ✅
            if hasattr(args, "seq_len"):
                model._modules[name].seq_len = args.seq_len
            
            model._modules[name].task = args.task
            if hasattr(args, "model_name_or_path"):
                model._modules[name].model_name_or_path = args.model_name_or_path

            # 🔍 DEBUG: Print args values (first layer only for verbose output)
            if layer_id == 31:  # layer_id starts at 32 and counts down
                print(f"\n{'='*80}")
                print(f"🔍 DEBUG: Entered isinstance block! Module type: {type(module).__name__}")
                print(f"🔍 DEBUG args - quest: {args.quest}, value_aware: {getattr(args, 'value_aware', 'N/A')}, fixed: {args.fixed}")
                print(f"{'='*80}\n")

            if args.quest and hasattr(args, "value_aware") and args.value_aware and getattr(args, "va_mode", "default") == "avg_v_l2norm":
                print(f"🔥 quest + VA enabled (avg_v_l2norm mode)") if layer_id == 2 else None
                model._modules[name].forward = types.MethodType(
                    forward_quest_avg_of_V_l2norm_group_consistent_sum, model._modules[name]
                )
            elif args.quest and hasattr(args, "value_aware") and args.value_aware:
                print(f"🔥 quest + value_aware enabled (✅ group consistent mode - SUM POOL)") if layer_id == 27 else None
                model._modules[name].forward = types.MethodType(
                    forward_quest_value_aware_group_consistent_sum, model._modules[name]
                )
            elif args.quest:
                print(f"🔥quest enabled ( ✅group consistent mode - SUM POOL) Layer: {layer_id}") # if layer_id == 2 else None
                model._modules[name].forward = types.MethodType(
                    # forward_quest, model._modules[name]
                    forward_quest_group_consistent_sum, model._modules[name]
                )
                if args.fixed:
                    model._modules[name].fixed = args.fixed
                    model._modules[name].mode = "quest_fixed"
                else:
                    model._modules[name].mode = "quest"
            else:
                print(f"🔥naive enabled") if layer_id == 2 else None
