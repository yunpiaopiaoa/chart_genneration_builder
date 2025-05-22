"""
待评估模型针对评测集生成推理答案，统一形成推理答案文件
后续答案文件将用于计算评测指标
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
import json
import logging
from pathlib import Path
import re

from langchain_openai import ChatOpenAI
from tqdm import tqdm

from src.build.img_gen.echarts_img_generator_multithread import (
    EchartsImgGeneratorMultiThread,
)
from src.infer.eval_dataset import EvalDataset, EvalSample
from src.datamodel.infer_result import InferResult, TaskResult
from src.infer.eval_gpt import EvalGpt
from src.infer.toeval_llm import BaseToEvalLLM
from src.utils.extract import extract_block

cur_file_name = Path(__file__).stem
logger = logging.getLogger(cur_file_name)
logger.setLevel(logging.ERROR)
handler = logging.FileHandler(f"log/{cur_file_name}.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def process_task(
    eval_llm: BaseToEvalLLM,
    task_name: str,
    messages,
    eval_messages,
    ground_truth,
    img_executor: ThreadPoolExecutor,
    chart_img_gen: EchartsImgGeneratorMultiThread,
    save_path: Path,
):
    # print("processing task", task_name, save_path)
    prediction = eval_llm.answer(eval_messages)
    if task_name.endswith("2code"):
        try:
            # code = extract_block(ts["prediction"])
            code = extract_block(prediction, "html")
            img_executor.submit(
                chart_img_gen.generate_img,
                code,
                save_path / f"{task_name}.png",
            )
        except Exception as e:
            logger.error(
                f"{task_name} failed to generate image {save_path / f"{task_name}.png"}, {e}"
            )
    return TaskResult(
        task=task_name,
        question=messages,
        ground_truth=ground_truth,
        prediction=prediction,
    )


def process_sample(
    eval_llm: BaseToEvalLLM,
    eval_sample: EvalSample,
    executor: ThreadPoolExecutor,
    img_executor:ThreadPoolExecutor,
    chart_img_gen: EchartsImgGeneratorMultiThread,
    save_path: Path,
    task_set:set[str]
):
    # print(f"Processing sample", save_path)
    futures = []
    task_results = []
    for (
        task_name,
        messages,
        eval_messages,
        ground_truth,
    ) in eval_sample.generate_task():
        if task_name not in task_set:
            continue
        future = executor.submit(
            process_task,
            eval_llm,
            task_name,
            messages,
            eval_messages,
            ground_truth,
            img_executor,
            chart_img_gen,
            save_path,
        )
        futures.append(future)
    for future in as_completed(futures):
        task_results.append(future.result())
    result = InferResult(
        chart_data=eval_sample.chart_data,
        code=eval_sample.code,
        image=eval_sample.img_path,
        task_results=task_results,
    )
    with open(save_path / "infer_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return result


def main(sample_dir: str, infer_dir: str,tasks:list[str]):
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
    task_set=set(tasks)
    eval_llm = EvalGpt(llm)
    eval_set = EvalDataset(cur_dir / sample_dir)
    chart_img_gen = EchartsImgGeneratorMultiThread()
    infer_dir_path = cur_dir / infer_dir
    infer_dir_path.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=2) as outter_excutor:
        with ThreadPoolExecutor(max_workers=8) as inner_excutor:
            with ThreadPoolExecutor(max_workers=8) as img_excutor:
                futures = []
                for index, eval_sample in enumerate(eval_set):
                    save_path = infer_dir_path / str(index)
                    save_path.mkdir(exist_ok=True)
                    if sum(f.is_file() for f in save_path.iterdir())==4:
                        continue
                    futures.append(
                        outter_excutor.submit(
                            process_sample,
                            eval_llm,
                            eval_sample,
                            inner_excutor,
                            img_excutor,
                            chart_img_gen,
                            save_path,
                            task_set
                        )
                    )
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing samples"
                ):
                    future.result()
        outter_excutor.shutdown(wait=True)
    chart_img_gen.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成推理答案文件")
    parser.add_argument("--sample_dir", type=str, help="评测集样本目录")
    parser.add_argument("--infer_dir", type=str, help="推理结果输出目录")
    parser.add_argument("--tasks",nargs='+', help="推理任务列表")
    args = parser.parse_args()

    main(args.sample_dir, args.infer_dir,args.tasks)
