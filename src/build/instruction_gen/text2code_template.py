import random
import re
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.datamodel.annotation import ChartData, InstructionData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Text2CodeTemplate(BaseInstructionTemplate):
    task = "text2code"

    def __init__(self, language: str, model: ChatOpenAI):
        super().__init__(language)
        self.templates = {
            "en": {
                "query": [
                    "Please generate {language} code based on the following chart: <description>"
                ],
            },
            "zh": {
                "query": [
                    "请根据以下图表描述生成{language}代码：<description>",
                ],
            },
        }
        self.constraints={
            "en":"Generate code using {code_language}.",
            "zh":"使用{code_language}生成代码。"
        }
        self.prompt = PromptTemplate.from_template(
            """
        我希望你扮演一位对图表展示有需求的用户，你的目标是让大模型根据你提供的图表需求描述生成图表代码。
        请你沉浸代入用户视角，模仿用户的口吻，以{language}语言为主要语言，提出一份尽可能简洁清晰的需求描述，要求生成{code_language}代码,使得下面提供的图表代码、图表数据、图表标题以及图表描述恰好满足你的需求。
        你提出的图表需求描述没有必要使用礼貌用词，而是应该尽可能接近真实人类的语言风格，避免使用过多的修饰词和复杂的句式，注意一定要提到要求生成{code_language}代码和运行环境。
        以下是图表相关信息（仅供参考，不可直接引用）：
        图表代码：{code}
        图表数据：{chart_data}
        图表标题：{title}
        图表描述：{description}
        """
        )


        self.chain = self.prompt | model

    def get_instance(self, chart_data: ChartData, code_data: CodeData):
        response = self.chain.invoke(
            {
                "language":self.language,
                "code_language": code_data["language"],
                "code": code_data["code"],
                "chart_data": chart_data["data"],
                "title":chart_data["title"],
                "description": chart_data["description"]
            }
        )
        query=response.content
        if not re.search(code_data["language"], query,re.IGNORECASE):#如果没有提及要求生成的代码语言，加上约束
            query=query+self.constraints[self.language].format(code_language=code_data["language"])
        answer = "<code>"
        messages: list[Message] = [
            {"role": "user", "content": [{"type": "text", "value": query}]},
            {"role": "assistant", "content": [{"type": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "messages": messages,
        }
        return instance