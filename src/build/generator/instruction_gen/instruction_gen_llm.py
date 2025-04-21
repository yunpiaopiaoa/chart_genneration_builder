from typing import Literal

from langchain_openai import ChatOpenAI
from src.datamodel.annotation import ChartData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import InstructionData
from src.build.generator.instruction_gen.code2data_template import Code2DataTemplate
from src.build.generator.instruction_gen.data2code_template import Data2CodeTemplate
from src.build.generator.instruction_gen.data2type_template import Data2TypeTemplate
from src.build.generator.instruction_gen.img2code_template import Img2CodeTemplate
from src.build.generator.instruction_gen.img2data_template import Img2DataTemplate
from src.build.generator.instruction_gen.img2text_template import Img2TextTemplate
from src.build.generator.instruction_gen.img2type_template import Img2TypeTemplate
from src.build.generator.instruction_gen.qa_template import QATemplate
from src.build.generator.instruction_gen.text2code_template import Text2CodeTemplate
from src.build.generator.instruction_gen.text2data_template import Text2DataTemplate
from .base_template import BaseInstructionTemplate


class InstructionGen:
    def __init__(self,llm:ChatOpenAI,language:Literal["zh","en"]):
        self.templates: list[BaseInstructionTemplate] = [
            Img2TypeTemplate(language),
            Img2CodeTemplate(language),
            Img2DataTemplate(language),
            Img2TextTemplate(language),
            # Data2TypeTemplate(language),
            Data2CodeTemplate(language),
            Code2DataTemplate(language),
            Text2CodeTemplate(language),
            Text2DataTemplate(language),
            QATemplate(language, llm)
    ]

    def generate_instruction(self, chart_data: ChartData, code_data: CodeData):
        instructions: list[InstructionData] = []
        for template in self.templates:
            try:#QATemplate可能生成指令失败
                instruction_data = template.get_instance(chart_data, code_data)
                instructions.append(instruction_data)
            except:
                pass
        return instructions