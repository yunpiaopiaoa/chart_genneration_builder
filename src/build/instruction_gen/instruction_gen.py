from concurrent.futures import ThreadPoolExecutor, as_completed
from math import e
from threading import Thread
from typing import Literal

from langchain_openai import ChatOpenAI
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
import weakref

class InstructionGen:
    def __init__(self, llm: ChatOpenAI, language: Literal["zh", "en"]):
        self.templates: list[BaseInstructionTemplate] = [
            Img2CodeTemplate(language),
            Img2DataTemplate(language),
            Data2CodeTemplate(language),
            Text2CodeTemplate(language, llm),
            QATemplate(language, llm),
        ]
        # 初始化线程池（建议4-8个worker）
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="InstGen_")
        
        # 使用弱引用finalizer确保资源释放
        self._finalizer = weakref.finalize(
            self, 
            self._cleanup_executor,
            self.executor
        )

    def _cleanup_executor(self, executor):
        """安全关闭线程池"""
        executor.shutdown(wait=False)
        print("ThreadPoolExecutor 已释放")

    def generate_instruction(self, chart_data: ChartData, code_data: CodeData):
        instructions: list[InstructionData] = []
        futures = {
            self.executor.submit(template.get_instance, chart_data, code_data): template
            for template in self.templates
        }

        for future in as_completed(futures):
            try:
                instructions.append(future.result())
            except Exception as e:
                template = futures[future]
                print(f"Error in {template.task}: {str(e)}")
        
        return instructions

    def close(self):
        """显式关闭方法"""
        if hasattr(self, '_finalizer') and self._finalizer.detach():
            self._cleanup_executor(self.executor)

    def __del__(self):
        """析构函数备用"""
        self.close()
