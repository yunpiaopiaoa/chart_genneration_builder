from abc import abstractmethod

from src.datamodel.annotation import ChartData, CodeData


class BaseCodeGenerator:
    def __init__(self):
        pass

    @abstractmethod
    def generate_code(self, chart_data:ChartData)->CodeData:
        pass
