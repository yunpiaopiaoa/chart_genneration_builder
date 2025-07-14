import argparse
import configparser
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
import numpy as np
import pandas as pd
from src.datamodel.infer_result import InferResult
from src.eval.eval_process import EvalProcess
from src.build.img_gen.echarts_img_generator import EchartsImgGenerator


def main(infer_dir: str, eval_dir: str):
    # 读取配置
    cur_dir = Path(__file__).resolve().parent
    config_path = cur_dir / "config" / "config.ini"
    con = configparser.ConfigParser()
    con.read(config_path, encoding="utf-8")
    config = con["judge_llm"]
    judge_llm = ChatOpenAI(
        base_url=config["base_url"],
        model=config["model"],
        api_key=config["api_key"],
        temperature=0,##确保评估稳定性
    )
    
    def all_infer_results(infer_dir: Path):
        for sub_dir in infer_dir.iterdir():
            with (sub_dir / "infer_result.json").open("r", encoding="utf-8") as f:
                infer_result:InferResult=json.load(f)
                yield infer_result,sub_dir

    infer_result_path = cur_dir / infer_dir
    eval_path = cur_dir / eval_dir
    eval_path.mkdir(parents=True, exist_ok=True)
    ep = EvalProcess(judge_llm)
    eval_results = ep.eval(all_infer_results(infer_result_path))

    # 保存评估结果
    with (eval_path / "eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)

    df = pd.json_normalize(eval_results, sep="->").T.reset_index()
    df[["Task", "Metric"]] = df["index"].str.split("->", expand=True)
    df = df.drop(columns=["index"])
    df[0] = df[0].apply(lambda x: [round(num, 2) for num in x])
    df[["Task", "Metric",0]].to_markdown(str(eval_path / "all.md"),floatfmt=".2f")
    df["mean"] = df[0].apply(np.mean)
    df["std"] = df[0].apply(np.std)
    df["count"] = df[0].apply(len)
    df = df.drop(columns=[0])
    df.to_markdown(str(eval_path / "summary.md"),floatfmt=".2f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate inference results.")
    parser.add_argument(
        "--infer_dir",
        type=str,
        default="results_infer",
        help="Directory containing inference results.",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="results_eval",
        help="Directory to save evaluation results.",
    )
    args = parser.parse_args()
    main(args.infer_dir, args.eval_dir)
