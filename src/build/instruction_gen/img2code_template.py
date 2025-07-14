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
                    "Generate a single {language} file to recreate the chart shown in the image.",
                    "Using the image as a reference, create a {language} implementation of the chart.",
                    "Convert the chart in this image into {language} code, including all necessary styles and scripts.",
                    "Write {language} code to implement the chart/diagram from the image.",
                    "Create a self-contained {language} file that visually reproduces the chart from the image.",
                    "Translate the chart in this image into a single {language} file.",
                    "Implement the chart from the image using {language}.",
                    "Develop a {language} file that accurately represents the chart in the image.",
                    "Generate {language} code to build the chart from this image.",
                    "Produce self-contained {language} source code that is a high-fidelity match of the chart in the image.",
                    # "Please generate {language} code based on the image:",
                    # "Please generate {language} code based on the image.",
                    # "Convert this image into {language} code.",
                    # "Write {language} code that implements what's shown in the image.",
                    # "Create {language} code from the visual content in this image.",
                    # "Translate the diagram in this image to {language} code.",
                    # "Implement the logic shown in the image using {language}.",
                    # "Develop {language} code that corresponds to this image.",
                    # "Generate executable {language} code based on this image.",
                    # "Produce {language} source code matching the image content."
                ],
            },
            "zh": {
                "query": [
                    "生成一个独立的 {language} 文件，以复现图片中显示的图表。",
                    "以图片为参考，创建一个图表的 {language} 实现。",
                    "将此图片中的图表转换为 {language} 代码，并包含所有必要的styles和scripts。",
                    "编写 {language} 代码来实现图片中的图表/示意图。",
                    "创建一个独立的 {language} 文件，用于复现图片中的图表。",
                    "将此图片中的图表转译成一个独立的 {language} 文件。",
                    "使用 {language} 来实现图片中的图表。",
                    "开发一个能准确表示图片中图表的 {language} 文件。",
                    "生成 {language} 代码来构建此图片中的图表。",
                    "生成一份独立的 {language} 源代码，要求高度匹配图片中的图表。",
                    # "请根据图片生成{language}代码:",
                    # "请根据图片生成{language}代码",
                    # "请将这张图片转换为{language}代码",
                    # "请编写实现图片内容的{language}代码",
                    # "请根据图片中的视觉内容创建{language}代码",
                    # "请将图片中的图表翻译为{language}代码",
                    # "请使用{language}实现图片中展示的逻辑",
                    # "请开发与图片对应的{language}代码",
                    # "请基于此图片生成可执行的{language}代码",
                    # "请生成与图片内容匹配的{language}源代码"
                ],
            },
        }


    def get_instance(self, chart_data: ChartData, code_data: CodeData,img_path:str):
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
            # {"role": "assistant", "content": [{"type": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "messages": messages,
        }
        return instance
