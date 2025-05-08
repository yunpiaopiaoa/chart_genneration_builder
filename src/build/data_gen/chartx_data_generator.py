import csv
import json
import os
from pathlib import Path
import random
from src.build.data_gen.base_data_generator import BaseDataGenerator
from src.datamodel.annotation import ChartData


class ChartxDataGenerator(BaseDataGenerator):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.dic = {}
        with (self.data_dir / "ChartX_annotation.json").open(
            "r", encoding="utf-8"
        ) as f:
            anno = json.load(f)
            self.dic = {a["imgname"]: a["description"]["output"] for a in anno}

    def generate_data(self):
        return self.random_select()

    def random_select(self) -> ChartData:
        """随机选择一个文件
        WARNING
        """
        select_dir = random.choice([d for d in self.data_dir.iterdir() if d.is_dir()])
        select_type = select_dir.stem
        csv_dir = select_dir / "csv"
        txt_dir = select_dir / "txt"
        sample_path = random.choice(list(csv_dir.iterdir()))
        titile_file_path = next(
            f for f in txt_dir.iterdir() if f.stem == sample_path.stem
        )
        # print(select_dir, sample_path, titile_file_path)
        chart_data = ChartData(type=select_type, description=self.dic[sample_path.stem])
        with sample_path.open("r", newline="") as file:
            reader = csv.reader(file)
            headers = next(reader)  # 读取表头
            chart_data_dict = {
                header: [] for header in headers
            }  # 初始化字典，键为表头，值为列表
            for row in reader:
                for header, value in zip(headers, row):
                    chart_data_dict[header].append(
                        value
                    )  # 将每一行的值添加到对应的列表中
            chart_data["data"] = chart_data_dict
        with titile_file_path.open("r", encoding="utf-8") as file:
            chart_data["title"] = file.read()
        return chart_data
