import random
from src.datamodel.annotation import ChartData, InstructionData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Img2TypeTemplate(BaseInstructionTemplate):
    task = "img2type"

    def __init__(self, language: str):
        super().__init__(language)
        self.templates = {
            "en": {
                "query": [
                    "What is the type of the chart?",
                    "What type of chart is this?",
                ],
                "constrain":"Please answer only the type of chart."
            },
            "zh": {
                "query": [
                    "该图表数据对应的图表类型是什么？",
                    "请问这张图的图表类型是什么？",
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
            {"role": "user", "contents": [{"modality": "text", "value": query},{"modality": "image", "value": "<image>"}]},
            {"role": "assistant", "contents": [{"modality": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "conversations": messages,
        }
        return instance
