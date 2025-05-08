import random
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
        # self.prompt = PromptTemplate.from_template(
        #     """
        # 我希望你扮演一位对图表展示有需求的用户。
        # 请你模拟用户视角，模仿用户的口吻，提出一份详细且字数适中的需求描述，要求生成{language}代码,使得我提供的图表代码、图表数据、图表标题以及图表描述恰好满足你的需求。
        # 需求描述的限制如下：
        # 1. 需求描述中要明确表达期望图表达成的展示目标，如对比数据、呈现趋势等；
        # 2. 说明对图表类型、元素（标题、图例、轴标签等）的要求；
        # 3. 强调对数据展示形式（如数值标注、颜色区分等）和样式风格（简约、科技感等）的偏好。
        # 以下是图表相关信息：
        # 图表代码：{code}
        # 图表数据：{chart_data}
        # 图表标题：{title}
        # 图表描述：{description}
        # """
        # )
        # self.prompt = PromptTemplate.from_template(
        #     """
        # 你是一位图表前端工程师。
        # 请结合下面提供的图表信息，输出一行口语化的自然语言作为需求描述，假装是通过这句需求描述得到的以下图表相关信息，要求生成{language}代码。
        # 以下是图表相关信息：
        # 图表代码：{code}
        # 图表数据：{chart_data}
        # 图表标题：{title}
        # 图表描述：{description}
        # """
        # )
        # self.prompt = PromptTemplate.from_template(
        #     """
        # 你是一位图表前端工程师。
        # 请根据以下图表的相关设置信息，用一句自然流畅、口语化的话描述该图表的需求背景和用途，要求模型生成图表的{language}代码。
        # 请避免直接在描述中重复提供的图表代码、数据、标题或描述内容，而是用更抽象、通俗的语言表达图表的目的和意义。
        
        # 以下是图表的相关信息：
        # 图表代码：{code}
        # 图表数据：{chart_data}
        # 图表标题：{title}
        # 图表描述：{description}
        # """
        # )
        self.prompt = PromptTemplate.from_template(
            """
        我希望你扮演一位对图表展示有需求的用户，你的目标是让大模型根据你提供的图表需求描述生成图表代码。
        请你沉浸代入用户视角，模仿用户的口吻，提出一份尽可能简洁清晰的需求描述，要求生成{language}代码,使得下面提供的图表代码、图表数据、图表标题以及图表描述恰好满足你的需求。
        你提出的图表需求描述没有必要使用礼貌用词，而是应该尽可能接近真实人类的语言风格，避免使用过多的修饰词和复杂的句式。
        以下是图表相关信息（仅供参考，不可直接引用）：
        图表代码：{code}
        图表数据：{chart_data}
        图表标题：{title}
        图表描述：{description}
        """
        )


        self.chain = self.prompt | model

    def get_instance(self, chart_data: ChartData, code_data: CodeData):
        # query_template = random.choice(self.templates[self.language]["query"])
        # query = query_template.format(language=code_data["language"])
        # answer = "<code>"
        # messages: list[Message] = [
        #     {"role": "user", "content": [{"type": "text", "value": query}]},
        #     {"role": "assistant", "content": [{"type": "text", "value": answer}]},
        # ]
        # instance: InstructionData = {
        #     "task": self.task,
        #     "messages": messages,
        # }
        # return instance
        response = self.chain.invoke(
            {
                "language": code_data["language"],
                "code": code_data["code"],
                "chart_data": chart_data["data"],
                "title":chart_data["title"],
                "description": chart_data["description"]
            }
        )
        query=response.content
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