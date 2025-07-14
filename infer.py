"""
待评估模型针对评测集生成推理答案，统一形成推理答案文件
后续答案文件将用于计算评测指标
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import configparser
import datetime
import json
import logging
import os
from pathlib import Path
import shutil

import aiofiles
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from tqdm import tqdm

from src.build.img_gen.echarts_img_generator_async import EchartsImgGeneratorAsync
from src.eval.eval_dataset import EvalDataset, EvalSample
from src.datamodel.infer_result import InferResult, TaskResult
from src.utils.extract import extract_block


async def main(
    sample_dir: Path, infer_dir_path: Path, eval_tasks: list[str], workers: int
):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M")
    log_path=Path(f"log/infer/{date_str}/{time_str}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cur_file_name = Path(__file__).stem
    logger = logging.getLogger(cur_file_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    eval_set = EvalDataset(sample_dir)
    chart_img_gen = EchartsImgGeneratorAsync()
    progress_bar = tqdm(total=len(eval_set), desc="推理任务", leave=True)
    task_set = set(eval_tasks)
    semaphore = asyncio.Semaphore(workers)

    async def handle_sample(eval_sample: EvalSample, save_path: Path):
        """单个样本的推理和图片渲染"""
        async def write_file_async(path: Path, content: str):
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(content)
        async with semaphore:
            query_tasks: list[asyncio.Task[BaseMessage]] = []
            save_tasks = []
            infer_result = InferResult(
                chart_data=eval_sample.chart_data,
                code=eval_sample.code,
                image=eval_sample.img_path,
                task_results=[],
            )
            for (
                task_name,
                messages,
                eval_messages,
            ) in eval_sample.generate_task():
                if task_set and task_name not in task_set:
                    continue
                # 为ainvoke添加超时控制
                timeout_seconds = 120  # 可根据需要调整
                async def ainvoke_with_timeout(eval_messages):
                    try:
                        return await asyncio.wait_for(llm.ainvoke(eval_messages), timeout=timeout_seconds)
                    except asyncio.TimeoutError:
                        logger.error(f"{save_path} Timeout in task {task_name}: invoke timeout after {timeout_seconds}s")
                        return None
                task = asyncio.create_task(ainvoke_with_timeout(eval_messages))
                task.task_name = task_name  # 附加元数据
                task.messages = messages
                task.eval_messages = eval_messages
                query_tasks.append(task)
            async for task in asyncio.as_completed(query_tasks):# 异步执行推理任务
                prediction = None
                try:
                    response = await task
                    if response is None:
                        logger.error(f"{save_path} error in task {task_name}\n{prediction}")
                        continue
                    task_name = task.task_name
                    messages = task.messages
                    eval_messages = task.eval_messages
                    prediction = response.content
                    if task_name.endswith("2code") or task_name == "multi_round":
                        code = extract_block(prediction, "html")
                        img_task=chart_img_gen.generate_img(
                            code, save_path / f"{task_name}.png"
                        )
                        save_tasks.append(img_task)
                        write_task = write_file_async(save_path / f"{task_name}.html", code)
                        save_tasks.append(write_task)
                    ts = TaskResult(
                        task=task_name,
                        question=messages,
                        prediction=prediction,
                    )
                    infer_result["task_results"].append(ts)
                except Exception as e:
                    logger.error(
                        f"{eval_sample.relative_path} error in task {task_name}\n{e}\n{prediction}"
                    )           
            async with aiofiles.open(
                save_path / "infer_result.json", "w", encoding="utf-8"
            ) as f:
                await f.write(json.dumps(infer_result, indent=4, ensure_ascii=False))
            await asyncio.to_thread(shutil.copy,eval_sample.img_path, save_path / Path(eval_sample.img_path).name)
            for ta in await asyncio.gather(*save_tasks, return_exceptions=True):
                if isinstance(ta, Exception):
                    logger.error(
                        f"{eval_sample.relative_path} error in generation\n{ta}"
                    )
            progress_bar.update()
            progress_bar.refresh()

    coroutines = []
    for eval_sample in eval_set:
        save_path = infer_dir_path / eval_sample.relative_path
        save_path.mkdir(exist_ok=True, parents=True)
        if sum(1 for f in save_path.iterdir() if f.is_file())==len(tasks)*2+2:
            progress_bar.update()
            continue
        co = handle_sample(eval_sample, save_path)
        coroutines.append(co)
    await asyncio.gather(*coroutines,return_exceptions=True)
    progress_bar.close()
    await chart_img_gen.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成推理答案文件")
    parser.add_argument("--sample_dir", type=str, help="评测集样本目录")
    parser.add_argument("--infer_dir", type=str, help="推理结果输出目录")
    parser.add_argument("--tasks", nargs="+", help="推理任务列表")
    parser.add_argument("--workers", default=8, type=int, help="工作线程数")
    args = parser.parse_args()
    tasks = (
        args.tasks
        if args.tasks
        else ["img2code", "data2code", "text2code", "multi_round"]
    )

    cur_dir = Path(__file__).resolve().parent
    config_path = cur_dir / "config" / "config.ini"
    con = configparser.ConfigParser()
    con.read(str(config_path), encoding="utf-8")
    config = con["infer_llm"]
    config['streaming']='true'  # 如果需要使用 enable_thinking，必须设置 streaming=True
    llm = ChatOpenAI(**config)
    infer_dir_path: Path = cur_dir / args.infer_dir
    infer_dir_path.mkdir(parents=True, exist_ok=True)

    # 记录运行参数
    params = {
        "sample_dir": args.sample_dir,
        "infer_dir": args.infer_dir,
        "tasks": tasks,
        "workers": args.workers,
        "infer_model": config["model"],
    }
    with open(infer_dir_path / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    asyncio.run(
        main(cur_dir / args.sample_dir, infer_dir_path, tasks, args.workers)
    )

