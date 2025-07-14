import logging
from pathlib import Path
import re
import textwrap
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.build.code_gen.html_template import HtmlTemplate
from src.datamodel.annotation import ChartData
from src.datamodel.annotation import CodeData
from src.build.code_gen.base_code_generator import BaseCodeGenerator
from src.utils.extract import extract_block

class EchartsHtmlGeneratorLLM(BaseCodeGenerator):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(f"log/{__name__}.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    def __init__(self, model:ChatOpenAI):
        script_template="""
            你是一个专业的前端工程师。请根据以下图表字典或列表对象生成相应的ECharts HTML代码:
            {chartdata}
            注意：
            1. topic表示图表所涉及的主题或话题；
            2. type表示ECharts支持的图表类型；
            3. title表示图表的标题；
            4. data是图表的数据；
            5. 请考虑图表的美观性和可读性，图表的标题和图例以及图表内容不要重叠和遮挡，如果图表的数值有实际意义应当带上准确的单位；
            请确保代码可以在浏览器中正确执行。
            """
        prompt: PromptTemplate = PromptTemplate.from_template(textwrap.dedent(script_template))
        self.chain = prompt | model
        self.html_template=HtmlTemplate()        

    def generate_code(self, chart_data: ChartData) -> CodeData:
        try:
            outputs = self.chain.invoke({"chartdata": chart_data})
            s=extract_block(outputs.content)
            # script="\n".join(" "*8 + line for line in outputs.content.splitlines())
            # code=self.html_template.instance(script)
            # compressed_html_code = re.sub(r"\s*\n\s*", "", code)
            codedata=CodeData(language="echarts HTML",code=s)
            return codedata
        except Exception as e:
            self.logger.error(f"Generate html code failed:{e}\ntext:{outputs.content}")
            raise e