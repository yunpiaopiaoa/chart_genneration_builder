import random
from src.datamodel.annotation import ChartData, CodeData, InstructionData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Text2DataTemplate(BaseInstructionTemplate):
    task = "text2data"

    def __init__(self, language: str):
        super().__init__(language)
        self.templates = {
            "en": {
                "query": [
                    "Please generate chart data based on the chart description:<description>"
                ],
                "constrain": "Please return the chart data in JSON format, with the column titles as keys and the column data as values.",
            },
            "zh": {
                "query": ["请根据图表描述生成图表数据：<description>"],
                "constrain": "返回JSON格式字典，字典的键值对的值是扁平的数据列表。",
            },
        }

    def get_instance(self, chart_data: ChartData, code_data: CodeData):
        query_template = random.choice(self.templates[self.language]["query"])
        query = query_template
        query += self.templates[self.language]["constrain"]
        answer = "<chart_data>"
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": query},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "messages": messages,
        }
        return instance
