import argparse
import configparser
from pathlib import Path
from langchain_openai import ChatOpenAI
from src.build.build_process import BuildProcess


def main(gen_count: int, sample_dir: str):
    cur_dir = Path(__file__).resolve().parent
    # 读取配置
    config_dir = cur_dir / "config"
    config_path = config_dir / "config.ini"
    con = configparser.ConfigParser()
    con.read(config_path, encoding="utf-8")
    llm = ChatOpenAI(**con["build_llm"])

    language = "zh"
    bp = BuildProcess(llm, language)
    limited_types = [
        "gantt",
        "rose",
        "tree",
    ]
    for type in limited_types:
        bp.build(gen_count, cur_dir / sample_dir, [type])
    bp.chart_img_gen.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build charts and annotations.")
    parser.add_argument(
        "--gen_count", type=int, default=2, help="Number of charts to generate."
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="sample",
        help="Directory to save the charts and annotations.",
    )
    args = parser.parse_args()

    main(args.gen_count, sample_dir=args.sample_dir)
