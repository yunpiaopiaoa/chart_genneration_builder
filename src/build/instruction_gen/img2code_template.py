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
                    "Convert this image into {language} code.",
                    "Write {language} code that implements what's shown in the image.",
                    "Create {language} code from the visual content in this image.",
                    "Translate the diagram in this image to {language} code.",
                    "Implement the logic shown in the image using {language}.",
                    "Develop {language} code that corresponds to this image.",
                    "Generate executable {language} code based on this image.",
                    "Produce {language} source code matching the image content."
                ],
            },
            "zh": {
                "query": [
                    "请根据图片生成{language}代码:",
                    "请根据图片生成{language}代码",
                    "请将这张图片转换为{language}代码",
                    "请编写实现图片内容的{language}代码",
                    "请根据图片中的视觉内容创建{language}代码",
                    "请将图片中的图表翻译为{language}代码",
                    "请使用{language}实现图片中展示的逻辑",
                    "请开发与图片对应的{language}代码",
                    "请基于此图片生成可执行的{language}代码",
                    "请生成与图片内容匹配的{language}源代码"
                ],
            },
        }


    def get_instance(self, chart_data: ChartData, code_data: CodeData):
        query_template = random.choice(self.templates[self.language]["query"])
        query = query_template.format(language=code_data["language"])
        answer = "<code>"
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": query},
                    {"type": "image", "value": "<image>"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "messages": messages,
        }
        return instance
