import argparse
import configparser
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
import numpy as np
import pandas as pd
from src.datamodel.infer_result import InferResult
from src.eval.eval_process import EvalProcess
from src.build.generator.img_gen.echarts_img_generator import EchartsImgGenerator


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

    infer_result_path = cur_dir / infer_dir / "infer_result.json"
    results: list[InferResult] = []
    with infer_result_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    eval_path = cur_dir / eval_dir
    eval_path.mkdir(parents=True, exist_ok=True)
    chart_img_gen = EchartsImgGenerator()
    ep = EvalProcess(judge_llm, chart_img_gen)
    eval_results = ep.eval(results, eval_path)
    chart_img_gen.cleanup()

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
