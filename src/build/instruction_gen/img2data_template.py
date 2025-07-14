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
                    "Extract the chart data from this image.",
                    "Could you parse the data in this image?",
                    "Identify and extract all data points from the image.",
                    "Read and extract the numerical data shown in this image.",
                    "Convert the visual data in this image into structured format.",
                    "Please retrieve all data values displayed in the image."
                ],
                "constrain": "Please return a JSON-formatted dictionary, where all keys correspond to flat lists of one-dimensional data.If you are unable to extract any data, return an empty dictionary.",
            },
            "zh": {
                "query": [
                    "请从图片中提取图表数据:",
                    "请从图片中提取图表数据，",
                    "解析图片中的图表数据。",
                    "请识别并提取图片中的所有数据点",
                    "请读取并提取图片中显示的数字数据。",
                    "请将图片中的可视化数据转换为结构化格式，",
                    "请提取图片中展示的所有数据值，",
                    "请从这张图片中获取图表数据，"
                ],
                "constrain": "请返回一个JSON格式的字典，要求所有键对应的值都是扁平化的一维数据列表。如果实在无法提取出数据，返回一个空字典。",
            },
        }


    def get_instance(self, chart_data: ChartData, code_data: CodeData,img_path:str):
        query_template = random.choice(self.templates[self.language]["query"])
        query = query_template
        query += self.templates[self.language]["constrain"]
        answer = "<chart_data>"
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
