
from pathlib import Path
from src.datamodel.annotation import ChartData
from src.datamodel.annotation import CodeData
from .base_code_generator import BaseCodeGenerator

class EchartsHtmlGenerator(BaseCodeGenerator):
    def __init__(self, template_dir):
        self.template_dir = template_dir

    def generate_code(self,data:ChartData)->CodeData:
        """返回生成代码对象（包含代码语言字段）
        根据指定模板生成 ECharts 图表的 HTML 代码
        1.图表数据计算其对应的图表类型，可能对应多个图表类型
        2.根据图表类型选择相应图表类型下的一个模板
        3.数据经过分段、填充等方法进行模板实例化
        """
        pass
    def select_template(self,chart_type:str)->str:
        """"""
        pass