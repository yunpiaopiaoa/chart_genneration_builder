import argparse
from asyncio import futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
import json
from math import inf
from pathlib import Path
from langchain_openai import ChatOpenAI
from narwhals import DataFrame
from tqdm import tqdm
from src.eval.eval_dataset import relative_paths
from src.eval.evaluation3 import EvalProcess
import pandas as pd

from src.utils.calculate_score import calculate_all

def main(infer_dir: str, eval_dir: str,eval_num:int,workers:int):
    # 读取配置
    cur_dir = Path(__file__).resolve().parent
    config_path = cur_dir / "config" / "config.ini"
    con = configparser.ConfigParser()
    con.read(config_path, encoding="utf-8")
    config = con["eval_llm"]
    eval_llm = ChatOpenAI(temperature=0, **config)

    eval_path = Path(eval_dir)
    eval_path.mkdir(exist_ok=True, parents=True)
    critic_dir = "src/eval/critic2"
    eval_process = EvalProcess(eval_llm, critic_dir=critic_dir)
    tasks = ["img2code", "data2code", "text2code", "multi_round"]

    infer_relative_dirs=list(relative_paths(Path(infer_dir)))
    if eval_num<inf:
        infer_relative_dirs=infer_relative_dirs[:eval_num]

    eval_res_list=[]
    futures=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for infer_relative_dir in infer_relative_dirs:
            target_path=Path(eval_dir) / f"{infer_relative_dir}.json"
            target_path.parent.mkdir(exist_ok=True, parents=True)
            future=executor.submit(eval_process.eval, Path(infer_dir)/infer_relative_dir,target_path,tasks)
            futures.append(future)
        for future in tqdm(as_completed(futures),total=len(futures), desc="评估任务"):
            eval_res_list.append(future.result())
        executor.shutdown(wait=True)
    print(f"评估结果数量：{len(eval_res_list)}")
    calculate_all(eval_path, eval_res_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估过程和结果分析")
    parser.add_argument("--infer_dir", type=str, help="推理结果目录路径")
    parser.add_argument("--eval_dir", type=str, help="评估结果保存目录路径")
    parser.add_argument("--eval_num", type=int, default=inf, help="评估样本数量")
    parser.add_argument("--workers", type=int,default=16, help="工作线程数")
    args = parser.parse_args()

    main(args.infer_dir, args.eval_dir,args.eval_num,args.workers)
