import random
from src.datamodel.annotation import ChartData, InstructionData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Text2CodeTemplate(BaseInstructionTemplate):
    task = "text2code"

    def __init__(self, language: str):
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
            {"role": "user", "contents": [{"modality": "text", "value": query}]},
            {"role": "assistant", "contents": [{"modality": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "conversations": messages,
        }
        return instance
