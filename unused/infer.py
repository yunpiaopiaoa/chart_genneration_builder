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
from langchain_core.messages import BaseMessage
from tqdm import tqdm

from src.build.img_gen.echarts_img_generator import (
    EchartsImgGenerator,
)
from src.eval.eval_dataset import EvalDataset
from src.datamodel.infer_result import InferResult, TaskResult
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
    config = con["infer_llm"]
    llm = ChatOpenAI(**config)
    # 定义待评测模型,加载评测集
    task_set = set(tasks) if tasks else set()
    print(task_set)
    eval_set = EvalDataset(cur_dir / sample_dir)
    chart_img_gen = EchartsImgGenerator()
    infer_dir_path = cur_dir / infer_dir
    infer_dir_path.mkdir(parents=True, exist_ok=True)

    # 记录运行参数
    params = {
        "sample_dir": sample_dir,
        "infer_dir": infer_dir,
        "tasks": tasks if tasks else "all",
        "workers": workers,
        "infer_model":config["model"],
    }
    with open(infer_dir_path / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)

    infer_result_dic: dict[Path, InferResult] = {}
    lock = threading.Lock()
    pending_tasks = {}
    task_done_event = threading.Event()
    progress_bar = tqdm(total=len(eval_set), desc="推理任务")

    with ThreadPoolExecutor(max_workers=workers) as img_executor, ThreadPoolExecutor(
        max_workers=workers
    ) as io_executor:
        def handle(future, save_path:Path, task_name, messages, ground_truth):
            global pending_tasks
            try:
                response:BaseMessage = future.result()
                prediction = response.content
                if task_name.endswith("2code") or task_name=="multi_round":
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
                    pending_tasks[save_path] -= 1
                    if pending_tasks[save_path] == 0:
                        img_executor.submit(
                            (save_path / "infer_result.json").write_text,
                            json.dumps(infer_result_dic[save_path], indent=4, ensure_ascii=False)
                        )
                        # 移除已完成样本的计数器
                        del pending_tasks[save_path]
                        progress_bar.update(1)
                        if progress_bar.n == progress_bar.total:
                            task_done_event.set()


        for eval_sample in eval_set:
            save_path = infer_dir_path / eval_sample.relative_path
            save_path.mkdir(exist_ok=True,parents=True)
            if sum(1 for f in save_path.iterdir())==5:
                progress_bar.update(1)
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
                    pending_tasks[save_path]= pending_tasks.get(save_path, 0) + 1
                try:
                    future = io_executor.submit(
                        lambda eval_messages=eval_messages: llm.invoke(eval_messages)
                    )
                    future.add_done_callback(
                        lambda f, save_path=save_path, task_name=task_name, messages=messages, ground_truth=ground_truth: handle(
                            f, save_path, task_name, messages, ground_truth
                        )
                    )
                except Exception as e:  # 捕获所有可能的异常
                    logger.error(f"Task failed: {task_name}, error: {str(e)}")
        task_done_event.wait()
        img_executor.shutdown(wait=True)
        io_executor.shutdown(wait=True)
    chart_img_gen.cleanup()
