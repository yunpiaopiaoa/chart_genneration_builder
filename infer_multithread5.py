"""
待评估模型针对评测集生成推理答案，统一形成推理答案文件
后续答案文件将用于计算评测指标
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import configparser
import json
import logging
from pathlib import Path
import threading

from langchain_openai import ChatOpenAI
from tqdm import tqdm

from src.build.img_gen.echarts_img_generator_multithread import (
    EchartsImgGeneratorMultiThread,
)
from src.infer.eval_dataset import EvalDataset
from src.datamodel.infer_result import InferResult, TaskResult
from src.infer.eval_gpt import EvalGpt
from src.utils.extract import extract_block

cur_file_name = Path(__file__).stem
logger = logging.getLogger(cur_file_name)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(f"log/{cur_file_name}.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成推理答案文件")
    parser.add_argument("--sample_dir", type=str, help="评测集样本目录")
    parser.add_argument("--infer_dir", type=str, help="推理结果输出目录")
    parser.add_argument("--tasks", nargs="+", help="推理任务列表")
    parser.add_argument("--workers",default=8, type=int, help="工作线程数")
    args = parser.parse_args()
    sample_dir = args.sample_dir
    infer_dir: str = args.infer_dir
    tasks = args.tasks
    workers = args.workers

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
    task_set = set(tasks) if tasks else set()
    eval_llm = EvalGpt(llm)
    eval_set = EvalDataset(cur_dir / sample_dir)
    chart_img_gen = EchartsImgGeneratorMultiThread()
    infer_dir_path = cur_dir / infer_dir
    infer_dir_path.mkdir(parents=True, exist_ok=True)

    infer_result_dic: dict[Path, InferResult] = {}
    lock = threading.Lock()
    pending_tasks = 0
    task_done_event = threading.Event()

    with ThreadPoolExecutor(max_workers=workers) as img_executor, ThreadPoolExecutor(
        max_workers=workers
    ) as io_executor:

        def handle(future, save_path, task_name, messages, ground_truth):
            global pending_tasks
            try:
                prediction = future.result()
                if task_name.endswith("2code"):
                    code = extract_block(prediction, "html")
                    img_executor.submit(
                        lambda code=code, save_path=save_path: chart_img_gen.generate_img(
                            code, save_path / f"{task_name}.png"
                        )
                    )
                ts = TaskResult(
                    task=task_name,
                    question=messages,
                    ground_truth=ground_truth,
                    prediction=prediction,
                )
                with lock:
                    infer_result_dic[save_path]["task_results"].append(ts)
            except Exception as e:  # 捕获所有可能的异常
                logger.error(f"Task failed: {task_name}, error: {str(e)}")
            finally:
                with lock:
                    pending_tasks -= 1
                    progress_bar.update(1)
                    if pending_tasks == 0:
                        task_done_event.set()  # 触发文件写入

        for index, eval_sample in enumerate(eval_set):
            save_path = infer_dir_path / str(index)
            save_path.mkdir(exist_ok=True)
            if sum(f.is_file() for f in save_path.iterdir()) == 4:
                continue
            infer_result_dic[save_path] = InferResult(
                chart_data=eval_sample.chart_data,
                code=eval_sample.code,
                image=eval_sample.img_path,
                task_results=[],
            )
            for (
                task_name,
                messages,
                eval_messages,
                ground_truth,
            ) in eval_sample.generate_task():
                if task_set and task_name not in task_set:
                    continue
                with lock:
                    pending_tasks += 1
                future = io_executor.submit(
                    lambda eval_messages=eval_messages: eval_llm.answer(eval_messages)
                )
                future.add_done_callback(
                    lambda f, save_path=save_path, task_name=task_name, messages=messages, ground_truth=ground_truth: handle(
                        f, save_path, task_name, messages, ground_truth
                    )
                )
        progress_bar = tqdm(total=pending_tasks, desc="推理任务")
        task_done_event.wait()
        print(f"推理任务完成, 开始写入文件...")
        for save_path, infer_result in infer_result_dic.items():
            io_executor.submit(
                (save_path / "infer_result.json").write_text,
                json.dumps(infer_result, indent=4, ensure_ascii=False),
            )
        io_executor.shutdown(wait=True)
        img_executor.shutdown(wait=True)
    chart_img_gen.cleanup()
