import random
from src.datamodel.annotation import ChartData, InstructionData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Img2TextTemplate(BaseInstructionTemplate):
    task = "img2text"

    def __init__(self, language: str):
        super().__init__(language)
        self.templates = {
            "en": {
                "query": [
                    "Please describe the chart based on the image.",
                    "Please provide a textual description of the chart in the image below:",
                ],
            },
            "zh": {
                "query": [
                    "根据图片给出图表的文本描述",
                    "给出以下图表图片的描述：",
                ],
            },
        }

    def get_instance(self, chart_data: ChartData, code_data: CodeData) :
        query_template = random.choice(self.templates[self.language]["query"])
        query = query_template
        # answer = chart_data["description"]
        answer="<description>"
        messages: list[Message] = [
            {"role": "user", "content": [{"type": "text", "value": query},{"type": "image", "value": "<image>"}]},
            {"role": "assistant", "content": [{"type": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "messages": messages,
        }
        return instance
