# Latency: AOA-KV kernel vs original Quest kernel

The accuracy side of this repo simulates sparse attention in PyTorch. Latency
cannot be read off that simulation — it needs the real FlashInfer-based kernels.
So the comparison is between two compiled trees:

| tree | what it is | label in results |
|---|---|---|
| `kernels/aoa_kv` | upstream Quest + the value-aware scoring commit | `AOA_KV` |
| `kernels/quest_orig` | upstream Quest, unmodified | `QUEST` |

See [KERNEL_DIFF.md](KERNEL_DIFF.md) for exactly what the one commit changes.

## Run it

```bash
conda activate aoakv
bash setup.sh --build-kernels          # once per machine (needs nvcc)
source env.sh

# 30-second check that both trees actually run, before downloading any weights
python latency/selftest_kernel.py --kernel aoa_kv
python latency/selftest_kernel.py --kernel quest_orig

bash scripts/run_latency.sh --model longchat-7b-v1.5-32k \
     --contexts 8192,16384 --budgets 1024 --gpu 0
```

`selftest_kernel.py` builds a tiny randomly-initialised llama-7B-shaped model in
memory and pushes a prefill plus a few decode steps through it. It proves the
extension loads and the kernels execute; it says nothing about speed.

Each row is appended to `$RESULTS_DIR/latency/latency.csv` and the comparison
table is printed at the end (also available later via
`python latency/compare.py --csv <csv>`):

```
 context  budget  page  QUEST prefill  AOA prefill      Δ%    QUEST dec/tok  AOA dec/tok      Δ%
    8192    1024    16         992.61      1039.89    +4.8           21.675       22.028    +1.6
```

CSV columns: `method, kernel_md5, gpu, model, context_len, token_budget,
page_size, decode_len, iteration, prefill_ms, prefill_std_ms,
decode_ms_per_token, decode_std_ms, timestamp`.

**MHA llama models only.** `quest/models/` wraps `LlamaForCausalLM`, and
`QuestAttention` appends K/V with `num_key_value_heads` into a cache the
controller sized for `num_attention_heads`. With a GQA model (llama-3.x,
qwen2.5, mistral) those shapes disagree, so the kernel path is restricted to
MHA checkpoints — `longchat-7b-v1.5-32k` (what the earlier measurements used),
`llama2-7b`, `vicuna-7b-v1.5-16k`. The accuracy side has no such restriction:
it runs every model in `config_models.sh`.

On top of that, `quest/ops/csrc/topk.cu:28` asserts `num_heads == 32`
(upstream's comment: *"Not necessary, but for Llama-7b"*), so the kernel path is
effectively llama-7B-shaped: 32 query heads, head_dim 128, MHA. A model with a
different head count fails with
`CHECK_EQ(32, num_heads) failed` the first time a decode step reaches top-k
selection — that is the assert, not a bug in your setup.

## Methodology

`latency/bench_latency.py` implements this; the choices are deliberate.

| choice | why |
|---|---|
| CUDA events + `torch.cuda.synchronize()` on both sides | upstream `bench_textgen.py` timed with `perf_counter` and no sync, so async launches leaked into the numbers |
| `--warmup 3` discarded, then `--iteration 10` averaged | first iterations pay allocator and autotune costs |
| random `inputs_embeds`, no tokenizer | kernel latency does not depend on token identity, and this keeps prefill length exact |
| `torch.cuda.empty_cache()` + `quest_clear()` per iteration | each iteration starts from the same allocator state |
| one GPU, exclusively (`--gpu`, and the runner warns if it is already in use) | a co-tenant process makes every number noise |
| md5 of the loaded `_kernels*.so` recorded in every row | the only way to prove which kernel produced a number |
| std reported alongside the mean | so a 2% difference can be judged against run-to-run spread |

## The kernel-loading trap

Both trees install a python package named `quest` containing an extension named
`_kernels*.so`. Whichever directory sits earlier on `sys.path` wins — and an
editable install or a stale `PYTHONPATH` can silently win over both. This has
already produced a measurement of a *third*, unrelated kernel build.

Two guards:

1. `bench_latency.py --kernel {aoa_kv,quest_orig}` puts the requested tree at the
   front of `sys.path` itself, then **asserts** that the imported `quest` package
   resolves inside that tree, and aborts if it does not.
2. `env.sh` warns when `PYTHONPATH` contains another checkout of this code.

If you ever bypass the script, set `PYTHONPATH=<tree>` explicitly and check the
md5 before trusting the output.

## Reference numbers

L40S (single GPU, fp16), `longchat-7b-v1.5-32k`, budget 1024, page 16,
decode_len 256, 10 iterations — measured 2026-05-25 on the original trees
(`AOA_KV` md5 `aeeec57b…`, `QUEST` md5 `4e81b725…`). Your own build will print
different md5s — they depend on the compiler and the target arch. What has to
hold is that the *two trees* produce different binaries, which `setup.sh`
checks explicitly:

| context | QUEST prefill (ms) | AOA-KV prefill (ms) | QUEST decode (ms/tok) | AOA-KV decode (ms/tok) |
|---:|---:|---:|---:|---:|
| 8192 | 992.61 | 1039.89 | 21.675 | 22.028 |
| 16384 | 2232.52 | 2274.64 | 22.449 | 23.128 |
| 32768 | 4810.93 | 4568.08 | 24.124 | 25.432 |

Read these as a sanity baseline for a rebuild, not as a published result: they
come from a single L40S, one model, one budget, and the prefill ordering flips
at 32k. A separate note from the same day recorded `torch.OutOfMemoryError` at
16k during prefill on one L40S (KV cache is ~1 MB/token, and prefill adds a
transient peak on top), so if a context length OOMs for you, that is expected
on 46 GB — use a bigger GPU or shorter context rather than assuming a bug.

## Optional: kernel-only micro-benchmarks

`setup.sh --with-kernel-bench` additionally builds the upstream nvbench
executables (`kernels/<tree>/kernels/build/`), which time individual kernels
rather than a decode step. They answer a different question than the E2E numbers
above: Quest selects only `token_budget` tokens per step, so a full-context
kernel sweep is not a predictor of end-to-end time. Keep the two apart when
reporting.
