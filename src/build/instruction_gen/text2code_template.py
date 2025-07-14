import json
import random
import re
from langchain_core.prompts import PromptTemplate,HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from langchain_openai import ChatOpenAI

from src.datamodel.annotation import ChartData, InstructionData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import Message
from src.utils.img_uploader import img2url
from .base_template import BaseInstructionTemplate

# txt_q="""我希望你扮演一位对图表展示有需求的用户，你的目标是让大模型根据你提供的图表需求描述生成图表代码。
# 请你沉浸代入用户视角，模仿用户的口吻，以{language}语言为主要语言，提出一份尽可能简洁清晰的需求描述，要求生成{code_language}代码,使得下面提供的图表类型、图表数据、图表标题以及图表描述恰好满足你的需求。
# 你提出的图表需求描述没有必要使用礼貌用词，而是应该尽可能接近真实人类的语言风格，避免使用过多的修饰词和复杂的句式，注意一定要提到要求生成{code_language}图表代码，直接输出自然语言需求描述。
# 注意，图表中不应包含图表的描述，需求中无需强调使用具体的JS库，需求中避免提到响应式设计。
# 以下是图表相关信息（仅供参考，不可直接引用）：
# chart data:{chart_data}.
# chart image:
# """
txt_q="""你的任务是扮演一个“指令生成器”。根据下面提供的结构化信息，你需要生成一段模拟真实用户向AI请求制作图表的自然语言指令。

图表结构化信息：
1. 图表类型：{chart_type}
2. 图表数据：{chart_data}
3. 图表标题：{chart_title}
4. 图表内容：{chart_description}
5. 图表渲染图：{image_url}

角色与需求描述指南：
语言：使用{language}进行描述。
语气：直接、简洁、自然，像真实用户一样。避免使用客套话、复杂句式或过多修饰。
核心要求：必须明确要求生成{code_language}代码。
内容：必须清晰地说明所需的图表类型、图表标题。此外也可以涉及图表的具体数据或趋势、色彩、字体字号、布局、宽高、比例、坐标轴、图例等细粒度要求。

关键输出限制：
你的最终输出只能是那段自然语言的用户需求描述，不要添加任何其他文字。
需求描述的措辞必须能让人理解：图表描述仅为背景信息，不应在最终的图表上展示出来。
需求描述中不得提及具体的JS库或响应式设计。
"""
class Text2CodeTemplate(BaseInstructionTemplate):
    task = "text2code"

    def __init__(self, language: str, model: ChatOpenAI):
        super().__init__(language)
        self.constraints={
            "en":"Generate code using {code_language}.",
            "zh":"使用{code_language}生成代码。"
        }
        self.prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template([
                {
                    "type": "text",
                    "text": txt_q
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "{image_url}"},
                },
            ]
        ),])
        # self.extra_constraints='All chart elements (including title) should be placed in the DOM element with id "chart".'


        self.chain = self.prompt | model

    def get_instance(self, chart_data: ChartData, code_data: CodeData,img_path:str):
        response = self.chain.invoke(
            {
                "language":self.language,
                "code_language": code_data["language"],
                "chart_type": chart_data["type"],
                "chart_data": json.dumps(chart_data["data"],ensure_ascii=False),
                "chart_title":chart_data["title"],
                "chart_description": chart_data["description"],
                "image_url": img2url(img_path),
            }
        )
        query=response.content
        if not re.search(code_data["language"], query,re.IGNORECASE):#如果没有提及要求生成的代码语言，加上约束
            query=query+self.constraints[self.language].format(code_language=code_data["language"])
        answer = "<code>"
        messages: list[Message] = [
            {"role": "user", "content": [{"type": "text", "value": query}]},
            # {"role": "assistant", "content": [{"type": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "messages": messages,
        }
        return instance