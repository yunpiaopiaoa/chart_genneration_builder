from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from langchain_openai import ChatOpenAI
from src.build.instruction_gen.multiround_template import MultiRoundTemplate
from src.datamodel.annotation import ChartData
from src.datamodel.annotation import CodeData
from src.datamodel.annotation import InstructionData
from src.build.instruction_gen.code2data_template import Code2DataTemplate
from src.build.instruction_gen.data2code_template import Data2CodeTemplate
from src.build.instruction_gen.data2type_template import Data2TypeTemplate
from src.build.instruction_gen.img2code_template import Img2CodeTemplate
from src.build.instruction_gen.img2data_template import Img2DataTemplate
from src.build.instruction_gen.img2text_template import Img2TextTemplate
from src.build.instruction_gen.img2type_template import Img2TypeTemplate
from src.build.instruction_gen.qa_template import QATemplate
from src.build.instruction_gen.text2code_template import Text2CodeTemplate
from src.build.instruction_gen.text2data_template import Text2DataTemplate
from .base_template import BaseInstructionTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed


class InstructionGen:
    def __init__(self, llm: ChatOpenAI, language: Literal["zh", "en"]):
        self.templates: list[BaseInstructionTemplate] = [
            Img2CodeTemplate(language),
            # Img2DataTemplate(language),
            Data2CodeTemplate(language),
            Text2CodeTemplate(language, llm),
            # QATemplate(language, llm),
            MultiRoundTemplate(language, llm),
        ]

    def generate_instruction(
        self,
        chart_data: ChartData,
        code_data: CodeData,
        img_path: str,
    ):
        instructions: list[InstructionData] = []
        for template in self.templates:
            try:
                instruction=template.get_instance(chart_data, code_data, img_path)
                instructions.append(instruction)
            except Exception as e:
                print(f"Error in {template.task}: {str(e)}")
        return instructions