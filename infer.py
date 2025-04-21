"""
待评估模型针对评测集生成推理答案，统一形成推理答案文件
后续答案文件将用于计算评测指标
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
import json
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.infer.eval_dataset import EvalDataset
from src.datamodel.infer_result import InferResult, TaskResult
from src.infer.eval_gpt import EvalGpt
from src.infer.toeval_llm import BaseToEvalLLM

def process_task(eval_llm:BaseToEvalLLM,task_name, messages, eval_messages, ground_truth):
    prediction = eval_llm.answer(eval_messages)
    return TaskResult(
        task=task_name,
        query=messages,
        ground_truth=ground_truth,
        prediction=prediction,
    )

def main(sample_dir: str, infer_dir: str):
    cur_dir = Path(__file__).resolve().parent
    config_path = cur_dir / "config" / "config.ini"
    con = configparser.ConfigParser()
    con.read(str(config_path), encoding="utf-8")
    config = con["eval_llm"]
    llm = ChatOpenAI(
        base_url=config["base_url"],
        model=config["model"],
        api_key=config["api_key"],
    )
    # 定义待评测模型,加载评测集
    eval_llm = EvalGpt(llm)
    eval_set = EvalDataset(cur_dir / sample_dir)
    results = []
    for eval_sample in eval_set:
        task_results=[]
        # for task_name, messages, eval_messages, ground_truth in eval_sample.generate_task():
        #     task_result = process_task(eval_llm, task_name, messages, eval_messages, ground_truth)
        #     task_results.append(task_result)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for task_name, messages, eval_messages, ground_truth in eval_sample.generate_task():
                future = executor.submit(process_task,eval_llm, task_name, messages, eval_messages, ground_truth)
                futures.append(future)

            for future in as_completed(futures):
                ts = future.result()
                task_results.append(ts)
        result = InferResult(
            chart_data=eval_sample.chart_data,
            img_path=eval_sample.img_path,
            task_results=task_results,
        )
        results.append(result)
        break  # WARNING:测试时使用
    infer_dir_path = cur_dir / infer_dir
    infer_dir_path.mkdir(parents=True, exist_ok=True)
    with (infer_dir_path / "infer_result.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成推理答案文件")
    parser.add_argument(
        "--sample_dir", type=str, default="sample", help="评测集样本目录"
    )
    parser.add_argument(
        "--infer_dir", type=str, default="results_infer", help="推理结果输出目录"
    )
    args = parser.parse_args()

    main(args.sample_dir, args.infer_dir)
