from typing import Literal, NotRequired, TypedDict

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

    language: Literal["echarts HTML","python"]
    code: str


class MessageContent(TypedDict):
    """消息内容"""

    type: str = Literal["text", "image"]# 支持文本或图像
    value: str  #type="image"时，值为本地图片路径或base64编码或在线地址


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


