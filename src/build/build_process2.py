from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import re
import json5
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.build.code_gen.html_template import HtmlTemplate
from src.build.img_gen.echarts_img_generator import EchartsImgGenerator
from src.datamodel.annotation import Annotation, ChartData, CodeData
from src.build.img_gen.base_img_generator import BaseImgGenerator
from src.build.instruction_gen.instruction_gen import InstructionGen
from src.utils.extract import extract_block
from src.datamodel.chart_type import CHARTTYPES


class BuildProcessForEchartsExample:
    """专门为echarts官网示例编写的构建评测集过程"""

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"log/{__name__}.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    def __init__(
        self,
        llm: ChatOpenAI,
        chart_img_gen: BaseImgGenerator,
        instruction_gen: InstructionGen,
    ):
        self.chart_img_gen = chart_img_gen
        self.instruction_gen = instruction_gen
        template = PromptTemplate.from_template(
            """
下面将提供一段 ECharts 图表的示例代码，示例代码中可能缺少图表标题，图表数据项可能没有实际意义。
我希望你能够修改和完善示例代码，使得图表数据更加符合实际，并保证图表标题完整。
要求如下：
1. 首先根据示例代码的图表类型生成一个合适的主题作为图表标题，然后根据该主题改写图表数据，使得图表内容具备实际意义；
2. 修改后的图表类型要与示例代码的图表类型一致且风格相似；
3. 修改后的图表布局美观，图表标题、图例和图表内容等图表元素之间无遮挡。
请以 JSON 标准格式的字典作为返回结果(以{languege}语言为主体，不要出现"`"字符)，包含以下字段：
code：原地修改后的图表代码，代码应该为字符串形式。
chart_data：修改后的图表数据，是一个字典，字典的键值对的值是扁平的数据列表。
title：图表标题。
description：图表描述。
type：图表类型。图表类型与示例代码的图表类型一致，必须严格从{chart_types}中选一个最合适的图表类型作为取值。
以下是参考代码：
{code}
"""
        )

        self.chain = template | llm
        self.html_template = HtmlTemplate()

    def build(self, data_path: Path, sample_dir: Path, language: str):
        index = -1
        with ThreadPoolExecutor(max_workers=4) as executor:
            for sub_dir in data_path.iterdir():
                for file in sub_dir.iterdir():
                    index += 1
                    if file.stem!="pie-nest":
                        continue
                    js_path = file / "main.js"
                    with open(js_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        # 检查第一行是否以 'import' 开头，如果是则删除
                    if lines and lines[0].strip().startswith("import"):
                        lines = lines[1:]
                    script = "".join(lines)
                    print(f"Processing {js_path} for sample {index}")
                    self.logger.info(f"Processing {js_path} for sample {index}")
                    retry = 3
                    gen_script = None
                    for _ in range(retry):
                        try:
                            gen_script, data, title, description, chart_type = (
                                self.gen_code_chartdata(script, language)
                            )
                            break
                        except Exception as e:
                            if isinstance(e, KeyboardInterrupt):
                                exit()
                            continue
                    if not gen_script:
                        self.logger.error(
                            f"Failed to generate code and chart data for {js_path}"
                        )
                        continue
                    code = self.html_template.instance(gen_script)
                    target_dir: Path = sample_dir / chart_type / f"{index}"
                    target_dir.mkdir(exist_ok=True, parents=True)
                    executor.submit(
                        (target_dir / "index.html").write_text, code, encoding="utf-8"
                    )
                    code_data = CodeData(language="echarts", code=code)
                    chart_data: ChartData = ChartData(
                        title=title, description=description, type=chart_type, data=data
                    )
                    instructions = self.instruction_gen.generate_instruction(
                        chart_data, code_data
                    )
                    annotation = Annotation(
                        chart=chart_data,
                        code=code_data,
                        instructions=instructions,
                    )
                    executor.submit(
                        (target_dir / "annotation.json").write_text,
                        json.dumps(annotation, ensure_ascii=False, indent=4),
                        encoding="utf-8",
                    )
                    self.chart_img_gen.generate_img(
                        code_data["code"], f"{target_dir}/chart.png"
                    )

    def gen_code_chartdata(self, code: str, language: str):
        """根据示例代码，修改得到新的代码和图表数据（通过大模型）"""
        response = self.chain.invoke(
            {"code": code, "chart_types": CHARTTYPES, "languege": language}
        )
        dic = json5.loads(extract_block(response.content))
        return (
            extract_block(dic["code"]),
            dic["chart_data"],
            dic["title"],
            dic["description"],
            dic["type"],
        )
