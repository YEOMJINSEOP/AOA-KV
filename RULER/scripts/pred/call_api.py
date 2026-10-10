# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
from tqdm import tqdm
from pathlib import Path
import traceback
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from compat.manifest import read_manifest

# *********** CLAUDE ADDED ************
# Attention output tracking
from evaluation.quest_attention import output_tracker
from collections import defaultdict
import numpy as np
import torch
# *********** CLAUDE ADDED ************

SERVER_TYPES = ('hf',)

class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

# Server
parser.add_argument("--server_type", default='hf', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo', 
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

# ***********************************************************************************************
parser.add_argument("--quest", type=lambda x: x.lower() == 'true', default=False, help="Enable quest attention (True/False)")
parser.add_argument("--token_budget", type=int, default=1024)
parser.add_argument("--chunk_size", type=int, default=16)

parser.add_argument("--seq_len", type=int, default=0, help="seq_len")

parser.add_argument("--fixed", type=lambda x: x.lower() == 'true', default=False, help="Enable quest + fixed attention (True/False)")

parser.add_argument("--value_aware", type=lambda x: x.lower() == 'true', default=False, help="Enable Value-aware Quest (True/False)")        

# ***********************************************************************************************

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
if args.server_type == 'hf':
    args.threads = 1

def get_llm(tokens_to_generate):
    if args.server_type == 'hf':
        from model_wrappers import HuggingFaceModel
        llm = HuggingFaceModel(
            args.model_name_or_path,
            # ***************************************
            args,
            # ***************************************
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
    
    else:
        raise RuntimeError(f'Unsupported server type {args.server_type}')

    return llm

# *********** CLAUDE ADDED ************
def get_method_name(args):
    """args로부터 method_name 결정"""
    if args.token_budget >= 999999:
        return "Full"

    method_name = "Quest"
    if args.fixed:
        method_name += "+fixed"
    if args.value_aware:
        method_name += "+VA"
    method_name += f"-{args.token_budget}-{args.chunk_size}"
    return method_name
# *********** CLAUDE ADDED ************

def main():
    start_time = time.time()

    # Attention-output (AOE) dumping is opt-in: it costs memory per decode step
    # and writes one .pt per sample. Accuracy runs leave it off.
    output_tracker["enabled"] = os.environ.get("AOE_DUMP", "0") == "1"
    output_tracker["data"].clear()

    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
    
    
    # suffixes = []
    # args.msb and suffixes.append(f'-msb')
    # args.quest and suffixes.append(f'-quest-{args.token_budget}-{args.chunk_size}')
    # args.kivi and suffixes.append(f'-kivi-{args.k_bits}-{args.v_bits}')
    # suffix_str = ''.join(suffixes)
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'
        
        
    print(f'✅Predict {args.task} \n"from" {task_file}\n"to" {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    ######################################################### 
    # # Load data
    # if os.path.exists(pred_file):
    #     pred_index = [sample['index'] for sample in read_manifest(pred_file)]
    #     data = [sample for sample in read_manifest(task_file) if sample['index'] not in pred_index]
    # else:
    #     data = read_manifest(task_file)
    data = read_manifest(task_file)
    #########################################################
    
    # Load api
    llm = get_llm(config['tokens_to_generate'])

    def get_output(idx_list, index_list, input_list, outputs_list, others_list, truncation_list, length_list):
        nonlocal llm

        while True:
            try:
                pred_list = llm.process_batch(
                    prompts=input_list, 
                    task_sample_idx=idx_list[0]
                )
                print(f"⭐pred_list: {pred_list}")
                break
            except Exception as e:
                traceback.print_exc()

        zipped_iter = zip(pred_list, idx_list, index_list, input_list,
                          outputs_list, others_list, truncation_list, length_list)

        for pred, idx, index, input, outputs, others, truncation, length in zipped_iter:
            if isinstance(pred['text'], str):
                pred_text = pred['text']
            elif len(pred['text']) > 0:
                pred_text = pred['text'][0]
            else:
                pred_text = ''

            outputs_parallel[idx] = {
                'index': index,
                'pred': pred_text,
                'input': input,
                'outputs': outputs,
                'others': others,
                'truncation': truncation,
                'length': length,
            }

    threads = []
    outputs_parallel = [{} for _ in range(len(data))]

    batched_data = []
    batch = []
    for idx, data_point in enumerate(data):
        
        data_point['idx'] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    if len(batch):
        batched_data.append(batch)

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        # the data is processed sequentially, so we can store the start and end of current processing window
        start_idx = 0  # window: [start_idx, end_idx]
        for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
            idx_list = [data_point['idx'] for data_point in batch]
            print(f"⭐task sample idx: {idx_list}")
            end_idx = idx_list[-1]  # the data in a batch is ordered

            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    idx_list=idx_list,
                    index_list=[data_point['index'] for data_point in batch],
                    input_list=[data_point['input'] for data_point in batch],
                    outputs_list=[data_point['outputs'] for data_point in batch],
                    others_list=[data_point.get('others', {}) for data_point in batch],
                    truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                    length_list=[data_point.get('length', -1) for data_point in batch],
                ),
            )
            thread.start()
            threads.append(thread)

            is_last_batch = (batch_idx == len(batched_data) - 1)

            if (len(threads) == args.threads) or is_last_batch:
                for thread in threads:
                    thread.join()
                threads = []

                # dump the results in current processing window on disk
                for idx in range(start_idx, end_idx + 1):
                    if len(outputs_parallel[idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[idx]) + '\n')

                start_idx = end_idx + 1

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")

    # *********** CLAUDE ADDED ************
    # Attention output 저장
    method_name = get_method_name(args)
    save_attention_outputs(args, method_name)
    # *********** CLAUDE ADDED ************

# *********** CLAUDE ADDED ************
def save_attention_outputs(args, method_name):
    """각 sample의 attention output 벡터를 .pt 파일로 저장"""
    if not output_tracker["enabled"] or not output_tracker["data"]:
        print("⚠️ No attention output data to save")
        return

    # model_name 추출 (e.g., mistral-7b-instruct-v0.2)
    model_name = os.path.basename(args.model_name_or_path)
    seq_len = args.seq_len if args.seq_len > 0 else "unknown"

    dump_root = os.environ.get(
        "AOE_DUMP_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "aoe"),
    )
    save_dir = os.path.join(dump_root, model_name, str(seq_len), method_name, args.task)
    os.makedirs(save_dir, exist_ok=True)

    for sample_idx, layer_data in output_tracker["data"].items():
        # layer_data: {layer_id: [tensor, tensor, ...]} -> {layer_id: stacked_tensor}
        save_data = {}
        for layer_id, tensors in layer_data.items():
            if len(tensors) > 0:
                save_data[layer_id] = torch.stack(tensors)  # (num_steps, num_heads, head_dim)

        save_path = f"{save_dir}/sample_{sample_idx}.pt"
        torch.save(save_data, save_path)

    print(f"✅ Attention outputs saved to: {save_dir}")
    print(f"   - {len(output_tracker['data'])} samples saved")

    # 다음 task를 위해 data 초기화
    output_tracker["data"].clear()
# *********** CLAUDE ADDED ************

if __name__ == '__main__':
    main()
