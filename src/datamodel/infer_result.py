from typing import NotRequired, TypedDict

from src.datamodel.annotation import ChartData, CodeData, Message
from src.datamodel.task_type import TaskType


class TaskResult(TypedDict):
    task: TaskType
    scene:NotRequired[str]#任务下的具体场景，非必须
    question: list[Message]
    ground_truth: Message
    prediction: str


class InferResult(TypedDict):
    chart_data: ChartData
    code:CodeData
    image: str
    task_results: list[TaskResult]
