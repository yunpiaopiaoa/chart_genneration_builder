import random
from src.datamodel.annotation import ChartData, CodeData, InstructionData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Img2DataTemplate(BaseInstructionTemplate):
    task = "img2data"

    def __init__(self, language: str):
        super().__init__(language)
        self.templates = {
            "en": {
                "query": [
                    "Please extract data from the image:",
                    "Please extract data from the image.",
                ],
                "constrain": "Please return the chart data in JSON format, with the column titles as keys and the column data as values. Do not use ``` or other code block formats.",
            },
            "zh": {
                "query": [
                    "请从图片中提取图表数据:",
                    "请从图片中提取图表数据",
                ],
"constrain": "返回JSON格式数据，键为列标题，值为列数据列表，不要使用```或其他代码块格式。",            },
        }

    def get_instance(self, chart_data: ChartData, code_data: CodeData):
        query_template = random.choice(self.templates[self.language]["query"])
        query=query_template
        query+=self.templates[self.language]["constrain"]
        answer = "<chart_data>"
        messages: list[Message] = [
            {
                "role": "user",
                "contents": [
                    {"modality": "text", "value": query},
                    {"modality": "image", "value": "<image>"},
                ],
            },
            {"role": "assistant", "contents": [{"modality": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "conversations": messages,
        }
        return instance
