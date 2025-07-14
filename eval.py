import argparse
import asyncio
import configparser
import json
from math import inf
from pathlib import Path
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from src.utils.calculate_score import calculate_all, calculate_all_with_weights
from src.eval.eval_dataset import relative_paths
from src.eval.evaluation6 import EvalProcess


async def main(infer_dir: str, eval_dir: str, eval_num: int, workers: int,eval_process:EvalProcess,tasks: list[str]):
    infer_relative_dirs = list(relative_paths(Path(infer_dir)))
    if eval_num < inf:
        infer_relative_dirs = infer_relative_dirs[:eval_num]

    eval_res_list = []
    semaphore = asyncio.Semaphore(workers)  # 控制最大并发数

    async def process_eval(infer_relative_dir):
        async with semaphore:  # 限制并发
            target_path = Path(eval_dir) / f"{infer_relative_dir}.json"
            target_path.parent.mkdir(exist_ok=True, parents=True)
            # 调用异步 eval
            result = await eval_process.eval(
                Path(infer_dir) / infer_relative_dir, target_path, tasks
            )
            return result

    # 创建异步任务列表
    coroutines = [
        process_eval(infer_relative_dir) for infer_relative_dir in infer_relative_dirs
    ]

    pbar= tqdm(total=len(coroutines), desc="评估任务",leave=True)  
    async for coro in asyncio.as_completed(coroutines):  # 按完成顺序获取
        res = await coro
        eval_res_list.append(res)
        pbar.update(1)
        pbar.refresh()
    # print(f"评估结果数量：{len(eval_res_list)}")
    calculate_all(eval_path , eval_res_list)
    calculate_all_with_weights(eval_path, eval_res_list)
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估过程和结果分析")
    parser.add_argument("--infer_dir", type=str, help="推理结果目录路径")
    parser.add_argument("--eval_dir", type=str, help="评估结果保存目录路径")
    parser.add_argument("--eval_num", type=int, default=inf, help="评估样本数量")
    parser.add_argument("--workers", default=16, type=int, help="工作线程数")
    args = parser.parse_args()

    # 读取配置
    cur_dir = Path(__file__).resolve().parent
    config_path = cur_dir / "config" / "config.ini"
    con = configparser.ConfigParser()
    con.read(config_path, encoding="utf-8")
    config = con["eval_llm"]
    eval_llm = ChatOpenAI(temperature=0, **config)
    print(f"eval model: {eval_llm.model_name}")
    tasks = ["img2code", "data2code", "text2code", "multi_round"]
    critic_dir = "src/eval/critic2_en"
    eval_process = EvalProcess(eval_llm, critic_dir=critic_dir)
    eval_path = Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True, parents=True)
    
    params = {
        "infer_dir": args.infer_dir,
        "eval_dir": args.eval_dir,
        "eval_num": args.eval_num if args.eval_num != inf else sum(1 for _ in relative_paths(Path(args.infer_dir))),
        "workers": args.workers,
        "model":config["model"],
        "temperature": eval_llm.temperature,
        "critic_dir": critic_dir,
        "tasks": tasks,
    }
    with open(Path(args.eval_dir) / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    asyncio.run(main(args.infer_dir, args.eval_dir, args.eval_num, args.workers,eval_process,tasks))
