# AOA-KV

Reproduction kit for **AOA-KV** (value-aware page scoring for Quest-style sparse
KV-cache attention), against **Quest** and dense attention. Two measurements,
nothing else:

| | what runs | needs CUDA toolkit? |
|---|---|---|
| **accuracy** | RULER + LongBench, attention simulated in PyTorch | no |
| **latency** | prefill / decode on the real FlashInfer kernels, AOA-KV tree vs original Quest tree | yes (`nvcc`) |

Everything is addressed relative to the repo root, so a freshly rented box needs
no path edits. No credentials are stored here.

## Quickstart on a fresh box

```bash
git clone <this repo> AOA-KV && cd AOA-KV
conda env create -f environment.yml && conda activate aoakv
bash setup.sh                       # pip layer + nltk data        -> accuracy ready
bash setup.sh --build-kernels       # + compile both kernel trees  -> latency ready
source env.sh
bash setup.sh --check               # verify everything
```

Gated models need `huggingface-cli login` (or `HF_TOKEN`). Weights and datasets
land in `$HF_HOME` — point it at a big disk, or at a pre-warmed cache, before
running anything:

```bash
export HF_HOME=/path/to/cache        # before: source env.sh
```

### Accuracy

```bash
bash scripts/run_ruler.sh      --model mistral-7b-instruct-v0.2 --method full  --gpu 0
bash scripts/run_ruler.sh      --model mistral-7b-instruct-v0.2 --method quest --gpu 0
bash scripts/run_ruler.sh      --model mistral-7b-instruct-v0.2 --method aoakv --gpu 0
bash scripts/run_longbench.sh  --model mistral-7b-instruct-v0.2 --method aoakv --gpu 0
```

Start small to check the plumbing before committing GPU hours:

```bash
bash scripts/run_ruler.sh --model llama-3.2-3b-instruct --method aoakv \
     --seq_lens 4096 --budgets 1024 --tasks niah_single_1 --num_samples 2 --gpu 0
```

### Latency

```bash
bash scripts/run_latency.sh --model longchat-7b-v1.5-32k \
     --contexts 8192,16384 --budgets 1024 --gpu 0
```

Prints an AOA-KV vs QUEST table and appends to
`$RESULTS_DIR/latency/latency.csv`, with the md5 of the kernel that produced
each row.

## Layout

```
env.sh                     source this first (paths, PYTHONPATH, HF_HOME, wandb off)
setup.sh                   deps, kernel builds, self-check
scripts/                   run_ruler.sh, run_longbench.sh, run_latency.sh
evaluation/                attention simulation + LongBench harness
  quest_attention_bench.py   the forward used by every accuracy run (BENCH_MODE=True)
  quest_attention.py         same formulas, non-bench variant
  llama.py mistral.py qwen.py  per-architecture KV-cache patches
  LongBench/                 pred.py, eval.py, metrics.py, config/
filtering/                 page scoring and top-k (filter_util_bench.py is the live one)
RULER/scripts/             data generation, prediction, scoring (NVIDIA RULER, trimmed)
  data/synthetic/json/       squad / hotpotqa / PaulGrahamEssays sources (vendored)
  config_models.sh           model registry + prompt template per model
compat/manifest.py         jsonl helpers that replace the NeMo dependency
kernels/aoa_kv/            Quest + value-aware scoring commit
kernels/quest_orig/        Quest, unmodified
latency/                   bench_latency.py (measure one tree), compare.py (table)
docs/                      ACCURACY.md, LATENCY.md, KERNEL_DIFF.md
results/                   outputs (gitignored)
third_party/               fetched by setup.sh (gitignored)
```

## Docs

- **[docs/ACCURACY.md](docs/ACCURACY.md)** — how to run RULER / LongBench, output
  layout, and the traps (BENCH_MODE, model-name spellings, AOE dumping, wandb).
- **[docs/LATENCY.md](docs/LATENCY.md)** — timing methodology, the
  kernel-loading trap, CSV schema, reference L40S numbers.
- **[docs/KERNEL_DIFF.md](docs/KERNEL_DIFF.md)** — the exact AOA-KV vs Quest
  kernel diff, the scoring formula, and where kernel and simulation disagree.

## Scope

Deliberately absent: ShadowKV and other baselines, AOE/error profiling, figure
generation, cluster (sbatch/docker) launchers, the `vllm`/`trtllm`/`sglang`
serving paths, and every non-`hf` RULER server type. Adding a baseline means
adding a `--method` arm to the two accuracy runners.

## Provenance

- Accuracy harness extracted from the VAKVPS research tree.
- `RULER/` is [NVIDIA RULER](https://github.com/NVIDIA/RULER) (Apache-2.0, see
  `RULER/LICENSE`), trimmed to the synthetic benchmark and the HF path, with the
  NeMo dependency replaced by `compat/manifest.py`.
- `kernels/*` are [mit-han-lab/quest](https://github.com/mit-han-lab/quest) at
  `01c1623` (see `kernels/*/LICENSE`); `kernels/aoa_kv` additionally carries the
  value-aware scoring commit.
- LongBench data is [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench).
