"""Run a few forward steps through a built kernel tree without downloading weights.

The kernel path is narrower than it looks: `quest/models/QuestAttention.py`
appends K/V with `num_key_value_heads` into a cache sized for
`num_attention_heads` (so MHA only), and `quest/ops/csrc/topk.cu:28` asserts
`num_heads == 32`. Finding that out after a 13 GB checkpoint download is a waste
of time, so this builds a tiny randomly-initialised llama-7B-shaped model in
memory and checks that prefill and decode actually run.

    python latency/selftest_kernel.py --kernel aoa_kv
    python latency/selftest_kernel.py --kernel quest_orig

It proves the extension loads and the kernels execute. It says nothing about
speed — the model is random and 4 layers deep.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_latency import KERNELS, load_kernel_tree  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=sorted(KERNELS), required=True)
    p.add_argument("--context_len", type=int, default=2048)
    p.add_argument("--decode_len", type=int, default=4)
    p.add_argument("--token_budget", type=int, default=256)
    p.add_argument("--page_size", type=int, default=16)
    args = p.parse_args()

    quest, md5 = load_kernel_tree(args.kernel)

    import torch
    from transformers import LlamaConfig

    device = torch.device("cuda:0")
    dtype = torch.float16
    torch.set_default_dtype(dtype)

    # llama-7B shaped where the kernels care: 32 heads, head_dim 128, MHA.
    cfg = LlamaConfig(
        hidden_size=4096, intermediate_size=4096, num_hidden_layers=4,
        num_attention_heads=32, num_key_value_heads=32,
        vocab_size=1000, max_position_embeddings=args.context_len + 512,
    )
    with device:
        model = quest.LlamaForCausalLM(cfg).to(dtype)
    model.quest_init(
        page_size=args.page_size,
        max_seq_len=args.context_len + args.decode_len + 512,
        token_budget=args.token_budget,
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        out = model(inputs_embeds=torch.randn(1, args.context_len, cfg.hidden_size,
                                              dtype=dtype, device=device))
        assert torch.isfinite(out.logits).all(), "prefill produced non-finite logits"
        for step in range(args.decode_len):
            out = model(inputs_embeds=torch.randn(1, 1, cfg.hidden_size,
                                                  dtype=dtype, device=device))
            assert torch.isfinite(out.logits).all(), f"decode step {step} produced non-finite logits"
    model.quest_clear()

    print(f"\nOK: {args.kernel} ran prefill ({args.context_len} tokens) + "
          f"{args.decode_len} decode steps, budget {args.token_budget}, md5 {md5[:8]}")


if __name__ == "__main__":
    main()
