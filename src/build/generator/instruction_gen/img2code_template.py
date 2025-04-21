import random
from src.datamodel.annotation import ChartData, CodeData, InstructionData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Img2CodeTemplate(BaseInstructionTemplate):
    task = "img2code"

    def __init__(self, language: str):
        super().__init__(language)
        self.templates = {
            "en": {
                "query": [
                    "Please generate {language} code based on the image:",
                    "Please generate {language} code based on the image.",
                ],
            },
            "zh": {
                "query": [
                    "请根据图片生成{language}代码:",
                    "请根据图片生成{language}代码",
                ],
            },
        }
        self.constraint = {
            "en": {
                "echarts": "Please directly generate html code, do not include unnecessary explanations and ``` characters, and ensure that it can run normally in the browser environment.",
                "python": "Please directly generate python code, do not include unnecessary explanations and ``` characters.",
            },
            "zh": {
                "echarts": "请直接生成html代码，不要使用```或其他代码块格式,确保能够在浏览器环境正常运行",
                "python": "请直接生成python代码，不要使用```或其他代码块格式。",
            },
        }

    def get_instance(self, chart_data: ChartData, code_data: CodeData):
        query_template = random.choice(self.templates[self.language]["query"])
        query = query_template.format(language=code_data["language"])
        query += self.constraint[self.language][code_data["language"]]
        answer = "<code>"
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
