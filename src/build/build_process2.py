import json
import logging
from pathlib import Path
import re
import json5
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.datamodel.annotation import Annotation, ChartData, CodeData
from src.build.img_gen.base_img_generator import BaseImgGenerator
from src.build.instruction_gen.instruction_gen import InstructionGen
from src.utils.extract import extract_block
from src.datamodel.chart_type import CHARTTYPES


class BuildProcessForEchartsExample:
    """专门为echarts官网示例编写的构建评测集过程"""

    def __init__(
        self,
        llm: ChatOpenAI,
        chart_img_gen: BaseImgGenerator,
        instruction_gen: InstructionGen,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        handler = logging.FileHandler(f"log/{__class__.__name__}.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.chart_img_gen = chart_img_gen
        self.instruction_gen = instruction_gen

        # template = PromptTemplate.from_template(
        #     """
        # 请你参考以下图表代码，修改成你自己的代码，并用你自己的图表数据进行替换。
        # 要求：
        # 1. 修改之后的图表数据要存在实际意义，与参考的图表类型一致；
        # 2. 保证代码渲染结果中的图表元素完整；
        # 3. 返回值为json格式字典，其中code字段为修改后的代码，chart_data字段为修改后的图表数据,title为图表标题，description为图表描述，type为图表类型。
        # 参考代码：{code}
        # """
        # )
        # template = PromptTemplate.from_template(
        # """
        # 以下echarts图表中的元素（标题）可能是完整的，也可能是不完整的，缺少某些图表的元素或者图表数据缺乏某种实际意义。
        # 现在请你修改代码，使得图表数据更加符合实际，并保证图表元素完整。另外，从修改后的图表代码中提取或总结出图表数据、图表标题、图表描述、图表类型。
        # 要求：
        # 1. 修改之后的图表与原来的图表类型一致，而且要与原来图表相似；
        # 2. 修改后的代码要能正确渲染出图表；
        # 3. 返回值为json格式字典，其中code字段为修改后的代码，chart_data字段为修改后的图表数据,title为图表标题，description为图表描述，type为图表类型；
        # 4.
        # 参考代码：{code}
        # """
        # )
        #
        #         template = PromptTemplate.from_template("""
        # 下面有一段 ECharts 图表示例代码，示例的items可能是由没有意义，我需要有具体含义的图表。
        # 首先你要根据js中的图表类型生成一个合适的主题作为图表标题，
        # 然后你需要根据这个主题改写出有意义的图表内容。
        # 确保图表中有标题，并且每一个数据项都是有实际意义的。
        # 按照示例的代码和风格，对图表内容对示例js代码做出调整，输出js代码,布局美观无遮挡。
        # 请以 JSON 格式的字典作为返回结果，包含以下字段：
        # code：修改后的图表代码。
        # chart_data：修改后的图表数据，为一个字典，字典的键值对的值是扁平的数据列表。
        # title：图表标题。
        # description：图表描述。
        # type：图表类型。
        # 以下是参考代码：
        # {code}
        # """)
        template = PromptTemplate.from_template(
            """
下面将提供一段 ECharts 图表的示例代码，示例代码中可能缺少图表标题，图表数据项可能没有实际意义。
我希望你能够修改和完善示例代码，使得图表数据更加符合实际，并保证图表标题完整。
要求如下：
1. 首先根据示例代码的图表类型生成一个合适的主题作为图表标题，然后根据该主题改写图表数据，使得图表内容具备实际意义；
2. 修改后的图表类型要与示例代码的图表类型一致且风格相似；
3. 修改后的图表布局美观，图表标题、图例和图表内容等图表元素之间无遮挡。
请以 JSON 格式的字典作为返回结果，包含以下字段：
code：修改后的图表代码。
chart_data：修改后的图表数据，为一个字典，字典的键值对的值是扁平的数据列表。
title：图表标题。
description：图表描述。
type：图表类型。图表类型与示例代码的图表类型一致，从{chart_types}中选一个图表类型作为取值。
以下是参考代码：
{code}
"""
        )

        self.chain = template | llm

    def build(self, gen_count: int, data_path: Path, sample_dir: Path):
        curdir = Path(__file__).parent
        with (curdir  / "code_gen" / "template.html").open(
            "r", encoding="utf-8"
        ) as f:
            self.html_template = f.read()
        index = 0
        for sub_dir in data_path.iterdir():
            for file in sub_dir.iterdir():
                target_dir = sample_dir / f"{index}"
                target_dir.mkdir(exist_ok=True, parents=True)
                js_path = file / "main.js"
                with open(js_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # 检查第一行是否以 'import' 开头，如果是则删除
                if lines and lines[0].strip().startswith("import"):
                    lines = lines[1:]
                script = "".join(lines)
                # if re.search(r"\$\.(getJSON|get)\(", script):
                #     ##跳过需要读取json文件的样例
                #     continue
                print(f"Processing {js_path}... for sample {index}")
                script, data, title, description, chart_type = self.gen_code_chartdata(
                    script
                )
                script = "\n".join(" " * 8 + line for line in script.splitlines())
                code = self.html_template.format(script=script)
                match = re.search(r'getElementById\((["\'])(.*?)\1\)', code)
                id = match.group(2)
                code = code.replace('<div id="chart"', f'<div id="{id}"')
                with open(f"{target_dir}/index.html", "w", encoding="utf-8") as f:
                    f.write(code)
                code_data = CodeData(language="echarts", code=code)
                chart_data: ChartData = ChartData(
                    title=title, description=description, type=chart_type, data=data
                )
                with open(f"{target_dir}/data.json", "w", encoding="utf-8") as f:
                    json.dump(chart_data, f, ensure_ascii=False, indent=4)
                instructions = self.instruction_gen.generate_instruction(
                    chart_data, code_data
                )
                annotation = Annotation(
                    chart=chart_data,
                    code=code_data,
                    # img_path=f"{target_dir.parent.stem}/{target_dir.stem}/chart.png",
                    instructions=instructions,
                )
                with open(f"{target_dir}/annotation.json", "w", encoding="utf-8") as f:
                    json.dump(annotation, f, ensure_ascii=False, indent=4)
                self.chart_img_gen.generate_img(
                    code_data["code"], f"{target_dir}/chart.png"
                )
                index += 1
                if index >= gen_count:
                    return

    def gen_code_chartdata(self, code: str):
        """根据示例代码，修改得到新的代码和图表数据（通过大模型）"""
        response = self.chain.invoke({"code": code, "chart_types": CHARTTYPES})
        dic = json5.loads(extract_block(response.content))
        return (
            extract_block(dic["code"]),
            dic["chart_data"],
            dic["title"],
            dic["description"],
            dic["type"],
        )
