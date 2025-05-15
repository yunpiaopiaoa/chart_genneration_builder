import base64
from io import BytesIO
from typing import Literal, NotRequired, TypedDict

from PIL import Image
from src.datamodel.task_type import TaskType


class ChartData(TypedDict):
    """图表数据
    data可视为二维表格数据字典key:[value]
    其中key为表头,value为key所在列的元素
    """

    title: str 
    description: str 
    type: str 
    data: dict 


class CodeData(TypedDict):
    """代码数据"""

    language: Literal["echarts,python"]
    code: str


# class TextContent(TypedDict):
#     type: Literal["text"]
#     text: str

# class ImageContent(TypedDict):
#     type: Literal["image_url"]
#     image_url: dict[str, str]  # 或进一步定义嵌套结构

# MessageContent = Union[TextContent, ImageContent]


class MessageContent(TypedDict):
    """消息内容"""

    type: str = Literal["text", "image"]# 支持文本或图像
    value: str  #type="image"时，值为base64编码


class Message(TypedDict):
    """对话消息
    一个角色的一条消息=若干个消息内容的拼接
    """

    role: Literal["user", "assistant"]
    content: str | list[MessageContent]


class InstructionData(TypedDict):
    """对话
    每个对话具备一个任务名称
    对话由对话消息组成
    """

    task: TaskType
    scene:NotRequired[str]#任务下的具体场景，非必须
    messages: list[Message]


class Annotation(TypedDict):
    chart: ChartData
    code: CodeData
    # img_path: str
    instructions: list[InstructionData]


def encode_base64(img_path: str):
    """返回图片的base64编码，附带data:image/jpeg;base64,前缀"""
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    res = "data:image/jpeg;base64,"
    res += base64.b64encode(buffered.getvalue()).decode("utf-8")
    return res
