import random
from src.datamodel.annotation import ChartData, InstructionData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Data2TypeTemplate(BaseInstructionTemplate):
    task = "data2type"

    def __init__(self, language: str):
        super().__init__(language)
        self.templates = {
            "en": {
                "query": [
                    "<chart_data>\nWhat is the type of the chart?",
                    "What type of chart is <chart_data>?",
                ],
                "constrain":"Please answer only the type of chart."
            },
            "zh": {
                "query": [
                    "<chart_data>\n该图表数据对应的图表类型是什么？",
                    "请问<chart_data>的图表类型是什么？",
                ],
                "constrain":"只需回答图表类型。"
            },
        }

    def get_instance(self, chart_data: ChartData, code_data: CodeData):
        query_template = random.choice(self.templates[self.language]["query"])
        query = query_template
        query+=self.templates[self.language]["constrain"]
        answer = chart_data["type"]
        messages: list[Message] = [
            {"role": "user", "contents": [{"modality": "text", "value": query}]},
            {"role": "assistant", "contents": [{"modality": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "conversations": messages,
        }
        return instance
