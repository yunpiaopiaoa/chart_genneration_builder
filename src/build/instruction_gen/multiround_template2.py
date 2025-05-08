import json
import logging
import random
import textwrap

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.build.instruction_gen.base_template import BaseInstructionTemplate
from src.datamodel.annotation import ChartData, CodeData, InstructionData, Message
from src.utils.extract import extract_block


class MultiRoundTemplate(BaseInstructionTemplate):
    task = "multi_round"

    def __init__(self, language: str, model: ChatOpenAI):
        super().__init__(language)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        handler = logging.FileHandler(f"log/{__class__.__name__}.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.model = model

    def get_instance(self, chart_data: ChartData, code_data: CodeData):
        history: list[BaseMessage] = []
        count = random.randint(2, 4)  # 随机对话轮数
        for i in range(count):
            if i == 0:
                question = self.generate_init_question(chart_data, code_data)
            else:
                question = self.generate_question(chart_data, code_data, history)
            history.append(HumanMessage(question))
            response = self.model.invoke(history)
            history.append(response)
        messages = []
        for index in range(0, len(history), 2):
            messages.append(Message(role="user", content=history[index].content))
            messages.append(
                Message(role="assistant", content=history[index + 1].content)
            )
        ins = InstructionData(task=self.task, messages=messages)
        return ins

    #TODO:修改模板使得大模型能够提出输入图片
    def generate_init_question(self, chart_data: ChartData, code_data: CodeData):
        """生成初始问题"""
        question = PromptTemplate.from_template(
            """
        请你生成一个有关echarts图表的生成需求。我将为你提供一些图表的参考信息。
        要求：
        1. 不需要完全参考图表信息；
        2. 需求描述尽量简洁明了，契合人类用户的语气口吻,避免使用过多的修饰词、礼貌用词和复杂句式；
        以下是图表相关信息（仅供参考，不可直接引用）：
        图表代码：{code}
        图表数据：{chart_data}
        图表标题：{title}
        图表描述：{description}
        """
        )
        promt_value = question.invoke(
            {
                "code": code_data["code"],
                "chart_data": json.dumps(chart_data["data"]),
                "title": chart_data["title"],
                "description": chart_data["description"],
            }
        )
        response = self.model.invoke(promt_value)
        return response.content

    def generate_question(
        self, chart_data: ChartData, code_data: CodeData, messages: list[BaseMessage]
    ) -> str:
        """基于对话历史生成后续问题的示例"""
        history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        sys_setting = """
        你是一位对图表展示有需求的前端工程师,善于通过不断地提出图表的编辑需求来优化图表效果。
        """
        prompt = PromptTemplate.from_template(
            """
        {sys}
        根据最近的对话历史：
        {history}
        请生成一个与图表编辑相关的后续问题，要求：
        1. 基于前文提到的修改需求,保持问题自然连贯；
        2. 沉浸代入用户视角，模仿用户的口吻，提出的问题尽可能简洁清晰；
        3. 提出的图表需求描述没有必要使用礼貌用词，而是应该尽可能接近真实人类的语言风格，避免使用过多的修饰词和复杂的句式。 
        """
        )
        prompt_value = prompt.format(sys=sys_setting, history=history)
        response = self.model.invoke(prompt_value)
        return response.content
