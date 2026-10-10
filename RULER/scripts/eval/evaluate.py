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
Get summary.csv with score and null predictions amount.

Running
```
python evaluate.py \
    --data_dir /path/to/your/prediction_jsonl_folder \
    --benchmark synthetic
```
"""

import re
import os
import argparse
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import pandas as pd
import importlib
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from compat.manifest import read_manifest, write_manifest
# ******** Claude ADDED ************
import wandb
# ******** Claude ADDED ************

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help='path to the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--verbose", type=int, default=0, help='how many lines you want to display.')
# ******** Claude ADDED ************
parser.add_argument("--model", type=str, default=None, help='Model name for wandb logging')
parser.add_argument("--method", type=str, default=None, help='Method name for wandb logging')
parser.add_argument("--token_budget", type=int, default=None, help='Token budget for wandb logging')
parser.add_argument("--max_seq_length", type=int, default=None, help='Max sequence length for wandb logging')
parser.add_argument("--bench_mode", type=str, default="False", help='Whether to log to wandb')
parser.add_argument("--page_size", type=int, default=None, help='Page size for wandb logging')
# ******** Claude ADDED ************
args = parser.parse_args()


def postprocess_pred(predict_str: str, task_config: dict):

    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def get_pred_and_ref(
    predictions_file: str,
    task_config: dict,
    input_field: str = 'input',
    references_field: str = 'outputs',
    prediction_field: str = 'pred',
    metadata_field: str = 'others',
):
    lines = read_manifest(predictions_file)

    inputs = []
    predicts = []
    references = []
    indices = []

    for line in tqdm(lines):
        input = line[input_field]
        predict = line[prediction_field]
        predict = postprocess_pred(predict, task_config)
        reference = line.get(references_field, [line.get('output', '')])
        index = line[metadata_field].get('id', line['index'])
        
        inputs.append(input)
        predicts.append(predict)
        references.append(reference)
        indices.append(index)
        
    return inputs, predicts, references, indices

def run_evaluation_per_task(task_config: dict, predictions_file: str, verbose: int = 1):
    inputs, predicts, references, indices = get_pred_and_ref(
        predictions_file=predictions_file,
        task_config=task_config,
    )

    task_nulls = f'{sum([len(x)==0 for x in predicts])}/{len(predicts)}'

    if len(references) > 0 and references[0][0] is not None:
        task_score = task_config['metric_fn'](predicts, references)
    else:
        task_score = 0.0

    if verbose != 0:
        print('=' * 40)
        for i, (input, reference, predict) in enumerate(zip(inputs, references, predicts)):
            # print(f'Input     : {input}')
            print(f'Reference : {reference}')
            print(f'Prediction: {predict}')
            print('=' * 40)
            if i > verbose:
                break
    print(f"🏴󠁧󠁢󠁳󠁣󠁴󠁿 {task_config['task']} score: {task_score}")
    return task_score, task_nulls, predicts, indices


def write_evaluation(results: dict):
    tasks = list(results.keys())
    score = [results[task]['score'] for task in tasks]
    nulls = [results[task]['nulls'] for task in tasks]
    dfs = [
        ['Tasks'] + tasks,
        ['Score'] + score,
        ['Nulls'] + nulls,
    ]

    output_file = os.path.join(args.data_dir, 'summary.csv' if len(tasks) > 1 else f'summary-{tasks[0]}.csv')
    df = pd.DataFrame(dfs)
    df.to_csv(output_file, index=False)
    print('\n=============================================\n')
    print(df)
    print(f'\nSaved eval results to {output_file}')


def write_submission(results: dict):
    COLUMNS = ["Task", "ID", "Prediction"]
    dfs = pd.DataFrame(columns=COLUMNS, data=[])
    
    for task, result in results.items():
        df = pd.DataFrame({
            'Task': task,
            'ID': result['indices'], 
            'Prediction': result['predicts']
        })
        dfs = pd.concat((dfs, df[COLUMNS]))
        
    output_file = os.path.join(args.data_dir, 'submission.csv')
    dfs = dfs.reset_index(drop=True)
    dfs.to_csv(output_file, index=False)
    print(f'\nSaved submission results to {output_file}')


def aggregate_chunk(folder):
    jsonl_files = [file for file in os.listdir(folder) if Path(file).suffix == '.jsonl' ]
    chunk_files = sorted([file for file in jsonl_files if re.match(r'.*[^_]+-\d+\.jsonl', file)])
    chunk_files_dict = defaultdict(list)
    for file in chunk_files:
        task = '-'.join(file.split('-')[:-1])
        chunk_files_dict[task].append(file)

    for task, files in chunk_files_dict.items():
        lines = []
        for file in sorted(files):
            file = os.path.join(folder, file)
            lines += read_manifest(file)
            os.remove(file) # Remove chunk files
        write_manifest(os.path.join(folder, f'{task}.jsonl'), lines)


def main():
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        module = importlib.import_module(f"{args.benchmark}.constants")
    except ImportError:
        print(f"Module eval.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        # synthetic.yaml
        tasks_customized = yaml.safe_load(f)

    
    TASKS = tasks_customized
    r"""
        [synthetic.yaml]
        TASKS: {        
            'niah_single_1': {
                'task': 'niah', 
                'args': {
                    'type_haystack': 'repeat', 
                    'type_needle_k': 'words',
                    'type_needle_v': 'numbers', 
                    'num_needle_k': 1, 
                    'num_needle_v': 1, 
                    'num_needle_q': 1
                    }
            }, 
            ...
        }        
    """
    
    for _, config in TASKS.items():
        config.update(tasks_base[config['task']])

    print(f"✅ Total tasks: {list(TASKS.keys())}")

    # Aggregate all prediction files
    aggregate_chunk(args.data_dir)

    # Get scores and nulls
    jsonl_files = [file for file in os.listdir(args.data_dir) if Path(file).suffix == '.jsonl']
    eval_results = {}
    subm_results = {}


    for task, config in TASKS.items():

        if f'{task}.jsonl' not in jsonl_files:
            print(f'Prediction file {task}.jsonl is not found.')
            continue

        print(f'✅ Evaluate task {task}...')
        task_score, task_nulls, predicts, indices = run_evaluation_per_task(
            predictions_file=os.path.join(args.data_dir, f'{task}.jsonl'),
            task_config=config,
        )
        eval_results[task] = {
            'score': task_score,
            'nulls': task_nulls,
        }
        subm_results[task] = {
            'predicts': predicts,
            'indices':indices,
        }
        
    # Write to csv
    write_evaluation(eval_results)
    write_submission(subm_results)

    # ******** Claude ADDED ************
    # Log to wandb if arguments are provided and bench_mode is True
    if args.model and args.method and args.token_budget is not None and args.bench_mode == "True":
        project_name = os.environ.get("WANDB_PROJECT", "aoakv-ruler-eval")
        # Run name includes page_size if provided
        run_name = str(args.token_budget)
        if args.page_size is not None:
            run_name = f"{args.token_budget}_p{args.page_size}"

        wandb.init(
            project=project_name,
            name=run_name,
            group=f"{args.model}_{args.method}_{args.max_seq_length}",
            tags=[args.model, args.method, str(args.max_seq_length or "")],
            config={
                "model": args.model,
                "method": args.method,
                "token_budget": args.token_budget,
                "max_seq_length": args.max_seq_length,
                "benchmark": args.benchmark,
            }
        )

        # Create table with task scores
        tasks = list(eval_results.keys())
        columns = ["token_budget"] + tasks + ["average"]
        table = wandb.Table(columns=columns)

        scores = [eval_results[task]['score'] for task in tasks]
        avg_score = sum(scores) / len(scores) if scores else 0
        row = [args.token_budget] + scores + [round(avg_score, 2)]
        table.add_data(*row)

        wandb.log({"results_table": table})
        wandb.finish()

        print(f"Results logged to wandb: {args.model}_{args.method}_{args.max_seq_length}, token_budget={args.token_budget}")
    # ******** Claude ADDED ************

if __name__ == '__main__':
    main()