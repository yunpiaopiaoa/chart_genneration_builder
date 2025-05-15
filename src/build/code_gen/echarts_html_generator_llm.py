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

class EchartsHtmlGeneratorLLM(BaseCodeGenerator):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(f"log/{__name__}.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    def __init__(self, model:ChatOpenAI):
        self.case_str=""
        curdir = Path(__file__).parent
        script_path = curdir / "script_example.js"
        with script_path.open("r", encoding="utf-8") as f:
            self.case_str = f.read()
        script_template="""
            你是一个专业的前端工程师。请根据以下图表字典或列表对象生成相应的ECharts脚本代码:
            {chartdata}
            注意：
            1. topic表示图表所涉及的主题或话题；
            2. type表示ECharts支持的图表类型；
            3. title表示图表的标题；
            4. data是图表的数据；
            5. 请考虑图表的美观性和可读性，图表的标题和图例以及图表内容不要重叠（可以通过设置echarts配置避免这一问题），如果图表的数值有实际意义应当带上准确的单位；
            6. 你生成的脚本代码将嵌入到html页面的srcipt标签内,我会提前引入ECharts库，所以不需要你使用import语句引入ECharts库；
            7. 你生成的脚本代码需要初始化ECharts实例挂载到id="chart"的dom元素上、配置图表属性、渲染图表，请确保每一步都正确执行；
            7. 生成结果除了ECharts脚本代码不要包含任何其他信息和多余字符，不要使用转义字符，开头结尾不要出现```等特殊符号，行首避免出现多余的空格和换行符。
            示例：
            {case_str}
            请生成类似的ECharts脚本代码，确保脚本代码可以在浏览器中正确执行。
            """
        prompt: PromptTemplate = PromptTemplate.from_template(textwrap.dedent(script_template))
        self.chain = prompt | model
        self.html_template=HtmlTemplate()        

    def generate_code(self, chart_data: ChartData) -> CodeData:
        try:
            outputs = self.chain.invoke({"case_str":self.case_str,"chartdata": chart_data})
            script="\n".join(" "*8 + line for line in outputs.content.splitlines())
            code=self.html_template.instance(script)
            compressed_html_code = re.sub(r"\s*\n\s*", "", code)
            codedata=CodeData(language="echarts",code=compressed_html_code)
            return codedata
        except Exception as e:
            self.logger.error(f"Generate html code failed:{e}")