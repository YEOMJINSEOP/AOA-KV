import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
# ****** Claude Added ******
import gc
from evaluation.quest_attention_bench import enable_quest_attention_eval
from evaluation.llama import enable_tuple_kv_cache_for_llama 
from evaluation.mistral import enable_tuple_kv_cache_for_mistral
from evaluation.qwen import enable_tuple_kv_cache_for_qwen2


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "mistral-7b-instruct-v0.3",
            "mistral-7b-instruct-v0.2",
            "llama-3.1-8b-instruct",
            "llama-3.2-3b-instruct", 
            "llama-3.2-1b-instruct",
            "qwen-2.5-7b-instruct",
            "qwen-2.5-3b-instruct",
        ],
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")

    parser.add_argument("--task", type=str, help="task name", required=True)

    # parser.add_argument("--token_budget", type=int, default=None)
    # parser.add_argument("--chunk_size", type=int, default=None)
    # parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")
    parser.add_argument("--cache_dir", type=str)
    
    parser.add_argument("--quest", type=lambda x: x.lower() == 'true', default=False, help="Enable quest attention (True/False)")
    parser.add_argument("--token_budget", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=16)
    
    parser.add_argument("--fixed", type=lambda x: x.lower() == 'true', default=False, help="Enable quest fixed attention (True/False)")
    parser.add_argument("--value_aware", type=lambda x: x.lower() == 'true', default=False, help="Enable quest fixed attention (True/False)")
    # ******** Claude ADDED ************
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for predictions")
    # ******** Claude ADDED ************

    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "llama-3" in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "qwen" in model_name:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        # split the prompt and question (simulate decoding in the question stage)
        if dataset in ["qasper", "hotpotqa", "2wikimqa", "musique"]:
            q_pos = prompt.rfind("Question:")
        elif dataset in ["multifieldqa_en", "gov_report", "qmsum", "multi_news"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        elif dataset in ["trec"]:
            q_pos = prompt.rfind("Please determine the type")
        elif dataset in ["samsum"]:
            q_pos = prompt.rfind("Summarize the dialogue")
        elif dataset in ["passage_retrieval_en"]:
            q_pos = prompt.rfind("Please enter")
        elif dataset in ["lcc", "repobench-p"]:
            q_pos = prompt.rfind("Please complete")
        else:
            q_pos = -1

        # max simulation length is 100
        q_pos = max(len(prompt) - 100, q_pos)

        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]

        if "chatglm3" in model_name:
            # input = prompt.to(device)
            input = prompt.to("cuda")
        else:
            # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            q_input = tokenizer(question, truncation=False, return_tensors="pt").to(
                "cuda"
            )
            q_input.input_ids = q_input.input_ids[:, 1:]

        context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

        # if (
        #     dataset == "samsum"
        # ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
        #     assert False
        #     output = model.generate(
        #         **input,
        #         max_new_tokens=max_gen,
        #         num_beams=1,
        #         do_sample=False,
        #         temperature=1.0,
        #         min_length=context_length + 1,
        #         eos_token_id=[
        #             tokenizer.eos_token_id,
        #             tokenizer.encode("\n", add_special_tokens=False)[-1],
        #         ],
        #     )[0]
        # else:
        # with torch.no_grad():
        with torch.inference_mode():
            output = model(
                input_ids=input.input_ids,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            for input_id in q_input.input_ids[0]:
                output = model(
                    input_ids=input_id.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = (
                    outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                )
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() == tokenizer.eos_token_id:
                    break

        # output = model.generate(
        #     **input,
        #     max_new_tokens=max_gen,
        #     num_beams=1,
        #     do_sample=False,
        #     temperature=1.0,
        # )[0]

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )

        # ****** Claude Added ******
        # Explicitly free memory after each sample to prevent OOM
        del past_key_values, output, input, q_input, pred_token_idx, generated_content, tokenized_prompt
        if 'outputs' in locals():
            del outputs
        torch.cuda.empty_cache()
        gc.collect()
        # ****** Claude Added ******

    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    if 'llama' in model_name.lower() or 'longchat' in model_name.lower():
        enable_tuple_kv_cache_for_llama()
    if 'mistral' in model_name.lower():
        enable_tuple_kv_cache_for_mistral()
    if 'qwen' in model_name.lower():
        enable_tuple_kv_cache_for_qwen2()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", cache_dir=args.cache_dir
    )
    model = model.eval()

    # ✅ original: if args.quest
    if args.quest:
        enable_quest_attention_eval(model, args)

    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name], 
        model_name, 
        device
    )
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper", # Single-doc QA, F1, avg 3k
            "multifieldqa_en", # Single-doc QA, F1, avg 4k
            "hotpotqa", # Multi-doc QA, F1, avg 9k
            "2wikimqa", # Multi-doc QA, F1, avg 4k
            "gov_report", # Summarization, Rouge-L, avg 8k
            "multi_news", # Summarization, Rouge-L, avg 2k
            "trec", # Few shot, Accuracy, avg 5k
            "triviaqa", # Few shot, F1, avg 8k
            "samsum", # Few shot, Route-L, 6k
            "passage_count", # Synthetic, Accuracy, 6k
            "passage_retrieval_en", # Synthetic, Accuracy, 9k
            "lcc", # Code, Edit Sim, 1k
            "repobench-p", # Code, Edit Sim, 4k
        ]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset("THUDM/LongBench", dataset, split="test", trust_remote_code=True)
            # ******** Claude ADDED ************
            out_dir = args.out_dir
            if args.quest:
                out_path = f"{out_dir}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"{out_dir}/{dataset}-full.jsonl"
            os.makedirs(out_dir, exist_ok=True)
            # ******** Claude ADDED ************
            print(f"output_path: {out_path}")          
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
