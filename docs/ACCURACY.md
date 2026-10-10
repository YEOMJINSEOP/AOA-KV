# Accuracy: RULER and LongBench

Three methods, one code path, flags are the only difference — that is what makes
the comparison fair:

| `--method` | what it is | flags | result-dir name |
|---|---|---|---|
| `full` | dense attention baseline | `quest=False` | `Full` |
| `quest` | Quest page selection | `quest=True fixed=True` | `Quest+fixed` |
| `aoakv` | Quest + value-aware scoring | `quest=True fixed=True value_aware=True` | `Quest+fixed+VA` |

`--fixed false` drops the fixed-selection term from both sparse methods (names
become `Quest` / `Quest+VA`). Keep it the same on both arms or the comparison
means nothing.

Attention is **simulated in PyTorch** here, not run through the CUDA kernels:
`evaluation/quest_attention_bench.py` replaces each attention module's
`forward`, and `filtering/filter_util_bench.py` does the page scoring and top-k.
Latency therefore cannot be read off these runs — see [LATENCY.md](LATENCY.md).

The scoring formula and its relationship to the kernel implementation is in
[KERNEL_DIFF.md](KERNEL_DIFF.md).

## RULER

```bash
conda activate aoakv && source env.sh
bash scripts/run_ruler.sh --model mistral-7b-instruct-v0.2 --method full  --gpu 0
bash scripts/run_ruler.sh --model mistral-7b-instruct-v0.2 --method quest --gpu 0
bash scripts/run_ruler.sh --model mistral-7b-instruct-v0.2 --method aoakv --gpu 0
```

Defaults: seq_lens `4096,8192,16384`, budgets `256,512,1024,2048`, page size 16,
all 13 synthetic tasks, 50 samples each (`config_tasks.sh`). Override with
`--seq_lens`, `--budgets`, `--tasks`, `--num_samples`, `--page_size`.

Three stages per run: `data/prepare.py` generates the synthetic data,
`pred/call_api.py` predicts, `eval/evaluate.py` scores. Prepared data is keyed
by model+seq_len and shared across methods and budgets, so only the first run
pays for it.

Output layout:

```
$RESULTS_DIR/ruler/<model>/synthetic/<seq_len>/
├── data/<task>/validation.jsonl              # shared across methods
├── Full/pred/                                # predictions + summary csv
└── Quest+fixed+VA-<budget>-<page>/pred/
```

All source data ships in the repo (`RULER/scripts/data/synthetic/json/`:
`squad.json`, `hotpotqa.json`, `PaulGrahamEssays.json`), so `qa_1`/`qa_2` work
offline too. `download_paulgraham_essay.py` is kept for provenance only — it
needs `bs4`/`html2text`, which this repo does not install.

Supported models are listed in `RULER/scripts/config_models.sh`. The template
type per model must not be changed: it decides the prompt wrapper, so editing one
makes new numbers incomparable with old ones.

## LongBench

```bash
bash scripts/run_longbench.sh --model mistral-7b-instruct-v0.2 --method aoakv --gpu 0
```

Defaults: 16 English tasks, budgets `256,512,1024,2048`. `pred.py` pulls the
datasets from the HF hub into `$HF_HOME` (≈110 MB) and writes one jsonl per
(task, method); `eval.py` then scores every task for one budget.

```
$RESULTS_DIR/longbench/<model>/<task>/<method>/
```

LongBench's own config keys spell three models differently from RULER
(`qwen-2.5-7b-instruct` vs `qwen2.5-7b-instruct`,
`longchat-v1.5-7b-32k` vs `longchat-7b-v1.5-32k`). `run_longbench.sh`
translates the RULER spelling, so pass the same `--model` value to both runners.

## Things that will bite you

- **`source env.sh` first.** `evaluation.*`, `filtering.*` and `compat.*` are
  imported by both benchmarks and need the repo root on `PYTHONPATH`. A stale
  `PYTHONPATH` pointing at another checkout of this code will be used instead —
  `env.sh` warns when it sees one.
- **`BENCH_MODE=True` (the default) selects the `*_bench` modules.** The
  published numbers were produced with those. `BENCH_MODE=False` switches to
  `evaluation/quest_attention.py` + `filtering/filter_util.py`, which implement
  the same formula but are not identical files — do not mix the two in one table.
- **wandb is off by default** (`WANDB_MODE=disabled`). To log:
  `export WANDB_MODE=online WANDB_API_KEY=... WANDB_PROJECT=...`. No key is
  stored in this repo; if you find one, treat it as a leak.
- **AOE dumping is off by default.** The research tree had
  `output_tracker["enabled"] = True` hardwired in `call_api.py`, which made every
  RULER run accumulate per-step attention outputs in memory and write `.pt`
  files. Here it is `AOE_DUMP=1` opt-in, written under `AOE_DUMP_DIR`.
- **GQA group-consistent selection** is part of the simulation (`logsumexp` over
  the query heads of a KV group before adding the value term), not of the CUDA
  kernel. See [KERNEL_DIFF.md](KERNEL_DIFF.md).
- **One unexposed variant exists.** `quest_attention_bench.py` also implements
  `forward_quest_avg_of_V_l2norm_group_consistent_sum`, reachable only if an
  `args.va_mode == "avg_v_l2norm"` attribute is set; no CLI flag sets it. It
  scores with the mean of per-token V norms instead of the norm of the mean V.
