import configparser
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from src.build.instruction_gen.multiround_template import MultiRoundTemplate
from src.datamodel.annotation import Annotation

if __name__ == "__main__":
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
    template = MultiRoundTemplate("echarts", llm)
    sample_path=Path("sample/sample27/1")
    with open(sample_path/"annotation.json", "r", encoding="utf-8") as f:
        anno:Annotation=json.load(f)
        chart_data=anno["chart"]
        code_data=anno["code"]
    ins=template.get_instance(chart_data, code_data)
    with open(sample_path/"multi_round_ins.json", "w", encoding="utf-8") as f:
        json.dump(ins, f, ensure_ascii=False, indent=4)