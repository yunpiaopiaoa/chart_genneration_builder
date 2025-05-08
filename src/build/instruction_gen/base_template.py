from src.datamodel.annotation import ChartData, InstructionData
from src.datamodel.annotation import CodeData
from src.datamodel.task_type import TaskType


class BaseInstructionTemplate:
    task: TaskType

    def __init__(self, language: str):
        self.language = language

    def get_instance(
        self, chart_data: ChartData, code_data: CodeData
    ) -> InstructionData:
        pass
