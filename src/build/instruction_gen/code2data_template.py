import random
from src.datamodel.annotation import ChartData, CodeData, InstructionData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Code2DataTemplate(BaseInstructionTemplate):
    task = "code2data"

    def __init__(self, language: str):
        super().__init__(language)
        self.templates = {
            "en": {
                "query": [
                    "Please extract the chart data from the following code snippet:<code>"
                ],
                "constrain": "Please return the chart data in JSON format, with the column titles as keys and the column data as values.",
            },
            "zh": {
                "query": ["请从以下代码片段中提取出图表数据：<code>"],
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
