import argparse
from concurrent.futures import ThreadPoolExecutor
import configparser
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from src.eval.evaluation import EvalProcess
import pandas as pd

def main(infer_dir: str, eval_dir: str):
    # 读取配置
    cur_dir = Path(__file__).resolve().parent
    config_path = cur_dir / "config" / "config.ini"
    con = configparser.ConfigParser()
    con.read(config_path, encoding="utf-8")
    config = con["judge_llm"]
    judge_llm = ChatOpenAI(temperature=0,**config)

    eval_path=Path(eval_dir)
    eval_path.mkdir(exist_ok=True,parents=True)
    critic_dir = "src/eval/critic3"
    eval_process = EvalProcess(judge_llm, critic_dir=critic_dir)
    tasks = ["img2code","data2code","text2code"]
    def handle(dir: Path):
        dic = eval_process.eval(dir, tasks)
        with (Path(eval_dir) / f"{dir.name}.json").open("w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)
        return dic

    with ThreadPoolExecutor(max_workers=8) as executor:
        eval_res_list = list(executor.map(handle, Path(infer_dir).iterdir()))

    # 将结果转换为 DataFrame
    score_dic = {"better": 1, "same": 0.5, "worse": 0}
    data = []
    for eval_dic in eval_res_list:
        for task, res in eval_dic.items():
            for k, v in res.items():
                data.append({
                    "task": task,
                    "metric": k,
                    "score":score_dic[v] if v in score_dic else v/5,
                })

    df = pd.DataFrame(data)
    # 计算各任务下的各指标的平均值，总体平均值（所有task的平均）
    # WARNING：不使用mean函数，未正确执行的评估任务评分为0，总数仍取样本数量
    analyze_res = df.groupby(["task", "metric"], as_index=False)["score"].sum()
    analyze_res["score"] = analyze_res["score"]/len(eval_res_list)
    overall_res = df.groupby("metric", as_index=False)["score"].sum()
    overall_res["score"] = overall_res["score"]/len(eval_res_list)*len(tasks)
    overall_res["task"] = "overall"  # 添加task标识
    # 合并结果输出文档
    final_res = pd.concat([analyze_res, overall_res], ignore_index=True)
    final_res['score'] = final_res['score'].round(5)
    final_res.to_markdown(eval_path / "analyze_res.md", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估过程和结果分析")
    parser.add_argument("--infer_dir", type=str, help="推理结果目录路径")
    parser.add_argument("--eval_dir", type=str, help="评估结果保存目录路径")
    args = parser.parse_args()

    main(args.infer_dir, args.eval_dir)
