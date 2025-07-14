import json
import logging
import re
import textwrap
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.datamodel.annotation import ChartData, InstructionData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import Message
from src.utils.extract import extract_block
from .base_template import BaseInstructionTemplate


class QATemplate(BaseInstructionTemplate):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(f"log/{__name__}.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    task = "qa"

    def __init__(self, language: str, model: ChatOpenAI):
        super().__init__(language)
        self.model = model
        template = """
            请你根据以下图表数据和代码，构造一个询问-回答对，用于验证你自己对图表的理解：
            图表数据：{chart_data}
            代码：{code}
            其中，代码会会完整包含图表数据的信息，并且会提供更多图表的布局和视觉信息。并且，代码的运行结果能够生成图表的图像。
            你可以考虑从图表数据提取、颜色和形状等视觉元素、图表描述和总结等角度去构造一个询问,并且自行给出准确的回答。
            示例：
            {case_str}
            请构造一个类似的询问-回答对，要求符合json格式，包含query和answer两个字段，字段值采用{language}语言,也不要返回多个json对象。
        """
        prompt = PromptTemplate.from_template(textwrap.dedent(template))
        self.chain = prompt | model
        self.case_str = json.dumps(
            {
                "query": "Create a brief summarization or extract key insights based on the chart.",
                "answer": "The chart shows the distribution of the number of sales per month for a company. ",
            }
        )
    
    def gen_message(self,query:str,answer:str,language:str):
        if language == "zh":
            messages: list[Message] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": "这是图表数据：<chart_data>"},
                        {"type": "text", "value": "这是代码：<code>"},
                        {"type": "image", "value": "这是图片：<image>"},
                        {"type": "text", "value": query},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": answer}],
                },
            ]
        elif language == "en":
            messages: list[Message] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": "This is the chart data: <chart_data>"},
                        {"type": "text", "value": "This is the code: <code>"},
                        {"type": "image", "value": "This is the image: <image>"},
                        {"type": "text", "value": query},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": answer}],
                },
            ]
        else:
            raise ValueError(f"Unsupported language: {language}")
        return messages

    def get_instance(self, chart_data: ChartData, code_data: CodeData,img_path:str):
        inputs = {
            "chart_data": chart_data,
            "code": code_data["code"],
            "case_str": self.case_str,
            "language": self.language,
        }
        outputs = self.chain.invoke(inputs)
        try:
            dic = json.loads(extract_block(outputs.content))
            query = dic["query"]
            answer = dic["answer"]

            instance: InstructionData = {
                "task": self.task,
                "messages": self.gen_message(query, answer, self.language)
            }
            return instance
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse JSON from output content: {outputs.content}"
            )
            raise e
