import argparse
import configparser
from pathlib import Path
from langchain_openai import ChatOpenAI
from src.build.build_process import BuildProcessForChartX
from src.build.data_gen.chartx_data_generator import ChartxDataGenerator
from src.build.code_gen.echarts_html_generator_llm import EchartsHtmlGeneratorLLM
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
    )

    # 定义生成器
    data_gen = ChartxDataGenerator(**dict(con["chartx_config"]))
    code_gen = EchartsHtmlGeneratorLLM(llm)
    chart_img_gen = EchartsImgGenerator()
    instruction_gen = InstructionGen(llm, language="zh")

    bp = BuildProcessForChartX(llm, data_gen, code_gen, chart_img_gen, instruction_gen)
    bp.build(gen_count, cur_dir / sample_dir)
    chart_img_gen.cleanup()

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
