"""Prefill / decode latency for one kernel tree (AOA-KV or original Quest).

The two kernel trees install the same python package name (`quest`) with the
same extension file name (`_kernels*.so`), so whichever one happens to be on
sys.path first is the one that gets measured. That has silently produced wrong
comparisons before (docs/LATENCY.md, "kernel loading trap"), so this script
takes `--kernel` and puts that tree at the front of sys.path itself, then
records the md5 of the extension it actually loaded next to every number.

Timing follows the methodology in docs/LATENCY.md: CUDA events with a
synchronize on both sides, `--warmup` iterations discarded, random embeddings
as input (the tokenizer plays no part in kernel latency).
"""

import argparse
import csv
import dataclasses
import hashlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS = {
    "aoa_kv": REPO_ROOT / "kernels" / "aoa_kv",
    "quest_orig": REPO_ROOT / "kernels" / "quest_orig",
}
# Label written to the CSV, so a row says which method it belongs to.
METHOD_LABEL = {"aoa_kv": "AOA_KV", "quest_orig": "QUEST"}

# MHA llama models only. quest/models/QuestAttention.py appends K/V with
# num_key_value_heads into a cache the controller allocated for num_attention
# heads, so a GQA model (llama-3.x, qwen2.5, mistral) does not work on this
# path — which is why the published latency numbers use longchat.
MODEL_PATHS = {
    "longchat-7b-v1.5-32k": "lmsys/longchat-7b-v1.5-32k",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "vicuna-7b-v1.5-16k": "lmsys/vicuna-7b-v1.5-16k",
}

CSV_FIELDS = [
    "method", "kernel_md5", "gpu", "model", "context_len", "token_budget",
    "page_size", "decode_len", "iteration", "prefill_ms", "prefill_std_ms",
    "decode_ms_per_token", "decode_std_ms", "timestamp",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", choices=sorted(KERNELS), required=True,
                   help="which vendored kernel tree to load and measure")
    p.add_argument("--model", default="longchat-7b-v1.5-32k",
                   help=f"one of {sorted(MODEL_PATHS)}, or pass --model_path")
    p.add_argument("--model_path", default=None, help="HF repo id or local dir")
    p.add_argument("--context_len", type=int, default=8192)
    p.add_argument("--decode_len", type=int, default=256)
    p.add_argument("--token_budget", type=int, default=1024)
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--iteration", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", default=None, help="CSV to append to (optional)")
    return p.parse_args()


def load_kernel_tree(kernel: str):
    """Put the requested tree first on sys.path and import `quest` from it."""
    tree = KERNELS[kernel]
    if not (tree / "quest" / "__init__.py").is_file():
        sys.exit(f"{tree} is not a kernel tree (quest/__init__.py missing)")
    sys.path.insert(0, str(tree))

    # `import quest` pulls in quest.utils, which imports the compiled extension
    # at module level — so an unbuilt tree fails right here, not at the explicit
    # import below.
    try:
        import quest  # noqa: E402  (must come after the sys.path edit)
    except ImportError as exc:
        if "_kernels" in str(exc):
            sys.exit(f"{tree} has no compiled extension yet ({exc}).\n"
                     f"Build both trees with:  bash setup.sh --build-kernels")
        raise

    loaded = Path(quest.__file__).resolve()
    if tree.resolve() not in loaded.parents:
        sys.exit(
            f"loaded `quest` from {loaded}, expected it under {tree}.\n"
            "Another copy is shadowing it — check PYTHONPATH and any editable installs."
        )
    try:
        import quest._kernels as _kernels
    except ImportError as exc:
        sys.exit(f"{tree} has no compiled extension yet ({exc}).\n"
                 f"Build it with:  bash setup.sh --build-kernels")
    so_path = Path(_kernels.__file__).resolve()
    md5 = hashlib.md5(so_path.read_bytes()).hexdigest()
    print(f"kernel tree : {tree}")
    print(f"quest pkg   : {loaded}")
    print(f"_kernels.so : {so_path}")
    print(f"md5         : {md5}")
    return quest, md5


def main():
    args = parse_args()
    quest, kernel_md5 = load_kernel_tree(args.kernel)

    import numpy as np
    import torch
    from tqdm.auto import tqdm

    model_path = args.model_path or MODEL_PATHS.get(args.model)
    if model_path is None:
        sys.exit(f"unknown --model {args.model}; pass --model_path instead")

    device = torch.device("cuda:0")
    dtype = torch.float16
    torch.set_default_dtype(dtype)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"gpu         : {gpu_name}")
    print(f"model       : {model_path}")

    with device:
        model = quest.LlamaForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype=dtype,
        )
    model.quest_init(
        page_size=args.page_size,
        max_seq_len=args.context_len + args.decode_len + 512,
        token_budget=args.token_budget,
        dtype=dtype,
        device=device,
    )
    hidden_size = model._config.hidden_size

    prefill_ms, decode_ms = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    total_iters = args.warmup + args.iteration
    with torch.inference_mode():
        for it in tqdm(range(total_iters), desc="iterations"):
            measured = it >= args.warmup
            torch.cuda.empty_cache()

            # ---- prefill ----
            hidden_states = torch.randn(1, args.context_len, hidden_size,
                                        dtype=dtype, device=device)
            torch.cuda.synchronize()
            start.record()
            model(inputs_embeds=hidden_states)
            end.record()
            torch.cuda.synchronize()
            if measured:
                prefill_ms.append(start.elapsed_time(end))

            # ---- decode ----
            for _ in range(args.decode_len):
                hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
                torch.cuda.synchronize()
                start.record()
                model(inputs_embeds=hidden_states)
                end.record()
                torch.cuda.synchronize()
                if measured:
                    decode_ms.append(start.elapsed_time(end))

            model.quest_clear()

    row = {
        "method": METHOD_LABEL[args.kernel],
        "kernel_md5": kernel_md5,
        "gpu": gpu_name,
        "model": model_path,
        "context_len": args.context_len,
        "token_budget": args.token_budget,
        "page_size": args.page_size,
        "decode_len": args.decode_len,
        "iteration": args.iteration,
        "prefill_ms": round(float(np.mean(prefill_ms)), 3),
        "prefill_std_ms": round(float(np.std(prefill_ms)), 3),
        "decode_ms_per_token": round(float(np.mean(decode_ms)), 4),
        "decode_std_ms": round(float(np.std(decode_ms)), 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print()
    print(",".join(CSV_FIELDS))
    print(",".join(str(row[k]) for k in CSV_FIELDS))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        new_file = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
