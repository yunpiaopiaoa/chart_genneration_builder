import argparse
import configparser
from pathlib import Path
from langchain_openai import ChatOpenAI
from src.build.build_process2 import BuildProcessForEchartsExample
from src.build.img_gen.echarts_img_generator import EchartsImgGenerator
from src.build.instruction_gen.instruction_gen import InstructionGen


def main(gen_count: int, sample_dir: str):
    cur_dir = Path(__file__).resolve().parent
    # 读取配置
    config_dir = cur_dir / "config"
    config_path = config_dir / "config.ini"
    con = configparser.ConfigParser()
    con.read(config_path, encoding="utf-8")
    llm = ChatOpenAI(
        base_url=con["build_llm"]["base_url"],
        model=con["build_llm"]["model"],
        api_key=con["build_llm"]["api_key"],
        temperature=0.6,
    )

    # 定义生成器
    chart_img_gen = EchartsImgGenerator()
    instruction_gen = InstructionGen(llm, language="zh")

    data_path = Path("echarts_examples")
    bp = BuildProcessForEchartsExample(llm, chart_img_gen, instruction_gen)
    bp.build(data_path, cur_dir / sample_dir,"chinese")
    chart_img_gen.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build charts and annotations.")
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="sample",
        help="Directory to save the charts and annotations.",
    )
    args = parser.parse_args()

    main(args.gen_count, sample_dir=args.sample_dir)
