from abc import abstractmethod

from src.datamodel.annotation import ChartData


class BaseDataGenerator:
    def __init__(self):
        pass

    @abstractmethod
    def generate_data(self) -> ChartData:
        pass
