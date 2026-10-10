import os
import json
import argparse
import numpy as np
# ******** Claude ADDED ************
import wandb
# ******** Claude ADDED ************

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    # ******** Claude ADDED ************
    parser.add_argument("--out_dir", type=str, required=True, help="Base output directory for predictions")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task names")
    parser.add_argument("--method", type=str, required=True, help="Method name (e.g., Quest, Quest+fixed, Quest+fixed+VA)")
    parser.add_argument("--token_budget", type=int, required=True, help="Token budget")
    parser.add_argument("--bench_mode", type=str, default="False", help="Whether to log to wandb")
    # ******** Claude ADDED ************

    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()

    # ******** Claude ADDED ************
    base_path = args.out_dir
    if not base_path.endswith('/'):
        base_path = base_path + '/'

    tasks = [x.strip() for x in args.tasks.split(',')]
    token_budget = args.token_budget

    # wandb 초기화 (bench_mode가 True일 때만)
    if args.bench_mode == "True":
        wandb.init(
            project="checking-longbench-eval",
            name=str(token_budget),
            group=f"{args.model}_{args.method}",
            tags=[args.model, args.method, str(token_budget)],
            config={
                "model": args.model,
                "method": args.method,
                "token_budget": token_budget,
            }
        )
        table = wandb.Table(columns=["task", "score"])

    results = {}
    scores_for_avg = []

    for task in tasks:
        # task별 경로 설정
        task_path = f"{base_path}{task}/{args.method}/"

        # 파일명 결정 (Full일 때는 {dataset}-full.jsonl)
        if args.method == "Full":
            if task.startswith('repobench-p'):
                filename = "repobench-p-full.jsonl"
            else:
                filename = f"{task}-full.jsonl"
        else:
            if task.startswith('repobench-p'):
                filename = f"repobench-p-{token_budget}.jsonl"
            else:
                filename = f"{task}-{token_budget}.jsonl"

        filepath = f"{task_path}{filename}"
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}, skipping...")
            continue

        print(f"Evaluating: {filepath}")

        predictions, answers, lengths = [], [], []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])

        if args.e:
            score = scorer_e(task, predictions, answers, lengths, all_classes)
        else:
            score = scorer(task, predictions, answers, all_classes)

        print(f"Task: {task}, Score: {score}")

        # 테이블에 row 추가 (bench_mode가 True일 때만)
        if args.bench_mode == "True":
            table.add_data(task, score)

        results[task] = score
        scores_for_avg.append(score)

    # AVG 계산 및 추가
    if scores_for_avg:
        avg_score = round(sum(scores_for_avg) / len(scores_for_avg), 2)
        print(f"AVG Score: {avg_score}")
        if args.bench_mode == "True":
            table.add_data("AVG", avg_score)
        results["AVG"] = avg_score

    # 테이블 로깅 (bench_mode가 True일 때만)
    if args.bench_mode == "True":
        wandb.log({"results_table": table})
        wandb.finish()
        print(f"Results logged to wandb: {args.model}_{args.method}, token_budget={token_budget}")

    # 결과 파일 저장
    out_path = f"{base_path}eval_results_{args.method}_{token_budget}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {out_path}")
    # ******** Claude ADDED ************
