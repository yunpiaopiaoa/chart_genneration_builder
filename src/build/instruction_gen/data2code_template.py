import random
from src.datamodel.annotation import ChartData, InstructionData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import Message
from .base_template import BaseInstructionTemplate


class Data2CodeTemplate(BaseInstructionTemplate):
    task = "data2code"

    def __init__(self, language: str):
        super().__init__(language)
        self.templates = {
            "en": {
                "query": [
                    "Generate a {language} chart from the following data:\n<chart_data>",
                    "Visualize this data as a self-contained {language} chart:\n<chart_data>",
                    "Create a {language} chart using the data below:\n<chart_data>",
                    "Convert this data into a {language} chart:\n<chart_data>",
                    "Generate a single {language} file to visualize this data.\n<chart_data>",
                    "Build a single-file {language} chart to represent the following data:\n<chart_data>",
                    "Write the {language} code to plot the following data as a chart:\n<chart_data>"
                    
                    # "please generate {language} code based on the following chart data: <chart_data>",
                    # "please generate {language} code based on the following chart data\n<chart_data>",
                    # "<chart_data>\nPlease generate {language} code based on the above chart data.",
                    # "Convert this chart data into {language} code:\n<chart_data>",
                    # "Write {language} code to visualize this data:\n<chart_data>",
                    # "Develop a {language} solution for the following data:\n<chart_data>",
                    # "Based on the data below, produce {language} code:\n<chart_data>",
                ],
            },
            "zh": {
                "query": [
                    "根据以下数据生成一个 {language} 图表：\n<chart_data>",
                    "将此数据可视化为一个独立的 {language} 图表：\n<chart_data>",
                    "使用下方的数据创建一个 {language} 图表：\n<chart_data>",
                    "将此数据转换为一个 {language} 图表：\n<chart_data>",
                    "生成一个独立的 {language} 文件来可视化此数据。\n<chart_data>",
                    "构建一个独立文件的 {language} 图表来呈现以下数据：\n<chart_data>",
                    "编写 {language} 代码，将以下数据绘制成一个图表：\n<chart_data>",

                    # "请根据以下图表数据生成{language}代码：<chart_data>\n",
                    # "请根据以下图表数据生成{language}代码\n<chart_data>\n",
                    # "<chart_data>\n请根据以上图表数据生成{language}代码。",
                    # "将这些图表数据转换为{language}代码：\n<chart_data>",
                    # "编写{language}代码来可视化这些数据：\n<chart_data>",
                    # "请为以下数据开发{language}解决方案：\n<chart_data>",
                    # "基于以下数据，生成{language}代码：\n<chart_data>",
                ],
            },
        }


    def get_instance(self, chart_data: ChartData, code_data: CodeData,img_path:str):
        query_template = random.choice(self.templates[self.language]["query"])
        query = query_template.format(language=code_data["language"])
        answer = "<code>"
        messages: list[Message] = [
            {"role": "user", "content": [{"type": "text", "value": query}]},
            # {"role": "assistant", "content": [{"type": "text", "value": answer}]},
        ]
        instance: InstructionData = {
            "task": self.task,
            "messages": messages,
        }
        return instance
