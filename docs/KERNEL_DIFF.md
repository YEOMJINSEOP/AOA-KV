# AOA-KV vs original Quest — what actually differs

Both `kernels/aoa_kv/` and `kernels/quest_orig/` are
[mit-han-lab/quest](https://github.com/mit-han-lab/quest) at upstream commit
`01c1623`. `aoa_kv` additionally carries one commit, *"VA kernel implemented"*
(`f6a85e6`, 2025-12-11). Nothing else separates them, which is what makes the
latency comparison meaningful: same prefill path, same page layout, same
top-k machinery, one changed scoring term.

Verify it yourself:

```bash
diff -rq kernels/quest_orig kernels/aoa_kv
```

| file | change |
|---|---|
| `kernels/include/decode/decode_attn.cuh` | page scoring gains the value term |
| `kernels/include/decode/decode_page.cuh` | append kernels compute `log‖V_avg‖₂` per page |
| `quest/ops/csrc/{bsk_ops.h,estimate.cu,page.cu}` | plumb the new buffer through the pybind layer |
| `quest/utils/{__init__.py,controller.py}` | allocate and pass the `v_log_norm` buffer |
| `quest/tests/test_*_claude.py` | three tests for the new scoring path (aoa_kv only) |

## The scoring change

Original Quest scores a page by the largest possible dot product between the
query and the page's min/max key bounds:

```
score = Σ_i max(q_i · k_max_i, q_i · k_min_i)
```

AOA-KV adds the page's value magnitude to that estimate
(`kernels/aoa_kv/kernels/include/decode/decode_attn.cuh:193`):

```cpp
// 2 * qK^T_approx / sqrt(head_dim) + 2 * log(||V_avg||_2)
float score = 2.f * qk_approx * rsqrt_head_dim + 2.f * v_log_norm_val;
```

`v_log_norm` is a `[num_layers, max_pages, num_heads]` fp16 buffer
(`quest/utils/controller.py`). The append kernels fill it the moment a page
becomes full — mean the page's V vectors, take the L2 norm, take the log with a
`1e-6` epsilon (`decode_page.cuh:501`, `:686`). The last, partially filled page
is never scored (it is always attended to), so it needs no entry.

## Same formula on the accuracy side

The accuracy numbers do not use these kernels. They come from the PyTorch
simulation in `filtering/filter_util.py`
(`local_heavy_hitter_mask_value_aware_group_consistent_sum`), which computes the
identical quantity:

```python
log_v_norm = torch.log(value_states_page_mean.norm(p=2, dim=-1) + 1e-8)   # :325
chunk_attn_weights = 2 * (chunk_attn_weights / math.sqrt(head_dim))       # :329
...
chunk_attn_weights = chunk_attn_weights + 2 * log_v_norm                  # :367
```

so `2·q̃K/√d + 2·log‖V_avg‖₂` either way (epsilon differs: `1e-8` in the
simulation, `1e-6` in the kernel — irrelevant at fp16 page norms).

## One real difference between kernel and simulation

**GQA aggregation.** The simulation is *group-consistent*: it `logsumexp`es the
scores of all query heads in a KV group before adding `2·log‖V‖`, so every head
in a group selects the same pages (`filter_util.py:361`). The kernel adds
`v_log_norm` per query head inside `compute_max_possible` and `decode_topk`
selects per head — there is no cross-head aggregation.

So the kernel implements AOA-KV's **scoring formula** but not its
**group-consistent selection**. For latency that is immaterial (identical work
per step). For accuracy it is not: do not expect the kernel path to reproduce
the simulation's scores. The accuracy harness in this repo deliberately uses
the simulation only.
