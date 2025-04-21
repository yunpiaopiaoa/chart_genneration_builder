import base64
from io import BytesIO
from typing import Literal, TypedDict

from PIL.ImageFile import ImageFile
from PIL import Image
from src.datamodel.task_type import TaskType


class ChartData(TypedDict):
    """图表数据
    data可视为二维表格数据字典key:[value]
    其中key为表头,value为key所在列的元素
    """

    title: str = "Chart Title"
    description: str = ""
    type: str = "unknown"
    data: dict = {}


class CodeData(TypedDict):
    """代码数据"""

    language: Literal["echarts,python"]
    code: str


class MessageContent(TypedDict):
    """消息内容"""

    modality: str = Literal["text", "image"]
    value: str | ImageFile  # 支持文本或图像


class Message(TypedDict):
    """对话消息
    一个角色的一条消息=若干个消息内容的拼接
    """

    role: Literal["user", "assistant"]
    contents: list[MessageContent]


class InstructionData(TypedDict):
    """对话
    每个对话具备一个任务名称
    对话由对话消息组成
    """

    task: TaskType
    conversations: list[Message]


class Annotation(TypedDict):
    chart: ChartData
    code: CodeData
    img_path: str
    instructions: list[InstructionData]


def encode_base64(img_path: str):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
