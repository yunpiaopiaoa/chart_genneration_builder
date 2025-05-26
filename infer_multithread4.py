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
import queue
import threading

from langchain_openai import ChatOpenAI

from src.build.img_gen.echarts_img_generator_multithread import (
    EchartsImgGeneratorMultiThread,
)
from src.infer.eval_dataset import EvalDataset
from src.datamodel.infer_result import InferResult, TaskResult
from src.infer.eval_gpt import EvalGpt
from src.utils.extract import extract_block

cur_file_name = Path(__file__).stem
logger = logging.getLogger(cur_file_name)
logger.setLevel(logging.ERROR)
handler = logging.FileHandler(f"log/{cur_file_name}.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成推理答案文件")
    parser.add_argument("--sample_dir", type=str, help="评测集样本目录")
    parser.add_argument("--infer_dir", type=str, help="推理结果输出目录")
    parser.add_argument("--tasks", nargs="+", help="推理任务列表")
    args = parser.parse_args()
    sample_dir = args.sample_dir
    infer_dir: str = args.infer_dir
    tasks = args.tasks

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

    task_queue = queue.Queue()
    img_queue = queue.Queue()
    infer_result_dic: dict[Path, InferResult] = {}
    lock = threading.Lock()
    stop_event = threading.Event()
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
            task_queue.put(
                (save_path, task_name, messages, eval_messages, ground_truth)
            )

    def process_task():
        while not task_queue and not stop_event.is_set():
            save_path, task_name, messages, eval_messages, ground_truth = (
                task_queue.get()
            )
            prediction = eval_llm.answer(eval_messages)
            if task_name.endswith("2code"):
                img_queue.put((save_path / f"{task_name}.png", prediction))
            ts = TaskResult(
                task=task_name,
                question=messages,
                ground_truth=ground_truth,
                prediction=prediction,
            )
            with lock:
                infer_result_dic[save_path]["task_results"].append(ts)
            task_queue.task_done()

    def process_img():
        while not img_queue and not stop_event.is_set():
            save_path, prediction = img_queue.get()
            code = extract_block(prediction, "html")
            chart_img_gen.generate_img(code, save_path)
            img_queue.task_done()

    for _ in range(8):
        threading.Thread(target=process_task).start()
    for _ in range(4):
        threading.Thread(target=process_img).start()
    task_queue.join()
    # 同时等待图片处理和文件写入完成
    with ThreadPoolExecutor(max_workers=4) as executor:
        for save_path, infer_result in infer_result_dic.items():
            executor.submit(
                (save_path / "infer_result.json").write_text,
                json.dumps(infer_result, indent=4, ensure_ascii=False),
            )
    img_queue.join()
    stop_event.set()
    chart_img_gen.cleanup()
