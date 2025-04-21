from typing import TypedDict

from src.datamodel.annotation import ChartData, Message
from src.datamodel.task_type import TaskType


class TaskResult(TypedDict):
    task: TaskType
    query: list[Message]
    ground_truth: Message
    prediction: str


class InferResult(TypedDict):
    chart_data: ChartData
    # code:CodeData #推理结果目前尚未需要code字段用于后续评估
    img_path: str
    task_results: list[TaskResult]
