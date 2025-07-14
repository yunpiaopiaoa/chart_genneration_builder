import json
import logging
import textwrap
import json5
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.datamodel.annotation import ChartData
from src.datamodel.chart_type import CHARTTYPES
from src.utils.extract import extract_block
from .base_data_generator import BaseDataGenerator



class LLMDataGenerator(BaseDataGenerator):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(f"log/{__name__}.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    def __init__(self,model:ChatOpenAI,language:str):
        self.language=language
        case = {
            "type": "折线图",
            "title": "近30年全球平均温度变化",
            "data": {
                "years": [1990, 1995, 2000, 2005, 2010, 2015, 2020],
                "temperatures": [14.6, 14.8, 15.0, 15.3, 15.7, 16.1, 16.5],
            },
        }
        template="""
            请根据以下模板生成一段JSON格式的图表数据。
            请以 JSON 标准格式的字典作为返回结果(以{languege}语言为主体，不要出现"`"字符)，包含以下字段：
            type：图表类型。严格从{chart_types}中选一个随机的图表类型作为取值。
            title：图表的标题，简洁明了，能够反映图表的主要信息；
            data：图表的数据，应符合所选图表类型的要求，并且数据应具有实际意义，格式上是一个字典，字典的键值对的值是扁平的数据列表。
            description：图表描述。
            示例：
            {case_str}
            请生成一个类似的JSON对象，数据可以涉及任何话题领域，但必须确保数据具有实际意义。
            同时，要保证数据足够精确，能够合理地支持所选的图表类型。
            """
        self.case_str = json.dumps(case, indent=None, ensure_ascii=False)
        prompt: PromptTemplate = PromptTemplate.from_template(textwrap.dedent(template))
        self.chain = prompt | model

    def generate_data(self,limited_types=CHARTTYPES) -> ChartData:
        outputs = self.chain.invoke({"case_str": self.case_str,"languege":self.language,"chart_types":limited_types if limited_types else CHARTTYPES})
        try:
            s=extract_block(outputs.content)
            data:ChartData = json5.loads(s)
            return data
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse JSON from output content: {outputs.content}"
            )
            raise e