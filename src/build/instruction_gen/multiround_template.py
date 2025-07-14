import json
import logging
import random
import textwrap

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.build.instruction_gen.base_template import BaseInstructionTemplate
from src.build.instruction_gen.data2code_template import Data2CodeTemplate
from src.build.instruction_gen.img2code_template import Img2CodeTemplate
from src.build.instruction_gen.text2code_template import Text2CodeTemplate
from src.datamodel.annotation import ChartData, CodeData, InstructionData, Message
from src.eval.eval_dataset import initialize_custom_fields
from src.utils.extract import extract_block


class MultiRoundTemplate(BaseInstructionTemplate):
    task = "multi_round"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(f"log/{__name__}.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    def __init__(self, language: str, model: ChatOpenAI):
        super().__init__(language)
        self.model = model
        self.init_templates:list[BaseInstructionTemplate]=[]
        img2code=Img2CodeTemplate(language)
        data2code=Data2CodeTemplate(language)
        text2code=Text2CodeTemplate(language,model)
        self.init_templates.append(img2code)
        self.init_templates.append(data2code)
        self.init_templates.append(text2code)
        self.scenes=[
            'chart title',
            'legend configuration',
            'color palette',
            'axis properties (range & labels)',
            'font settings (type & size)',
            'chart size & aspect ratio',
            'chart type',
            'chart layout',
                # "title content",
                # "legend content and position",
                # "single data and multiple tasks",
                # "graphic colors",
                # "axis range and name",
                # # "data display tips",
                # "font type and size",
                # "graphic dimensions",
                # "chart type or layout"
            ]
        

    def get_instance(self, chart_data: ChartData, code_data: CodeData,img_path:str):
        """该函数仅生成2轮对话，以第2轮的编辑需求作为scene值保存"""
        # history和save_history指代对话消息历史一致，save_history保留了待填充字段
        history: list[Message] = []
        save_history: list[Message]=[]
        selected:BaseInstructionTemplate=random.choice(self.init_templates)
        ins=selected.get_instance(chart_data, code_data,img_path)
        save_history.extend(ins["messages"])
        messages=initialize_custom_fields(ins["messages"],chart_data,code_data,img_path)
        history.extend(messages)
        
        answer_first_round=[{"role": "assistant", "content": [{"type": "text", "value": "<code>"}]}]
        save_history.extend(answer_first_round) #answer  in  first round.
        history.extend(initialize_custom_fields(answer_first_round,chart_data,code_data,img_path))
        
        scene=random.choice(self.scenes)
        question = self.generate_question(chart_data, code_data, history,scene)
        save_history.append(Message(role="user", content=[{"type": "text", "value":question}]))
        history.append(Message(role="user", content=question))
        # response = self.model.invoke(convert_messages_to_openai_message(history))
        # save_history.append(Message(role="assistant", content=[{"type": "text", "value":response.content}]))
        # history.append(Message(role="assistant", content=response.content))
        ins = InstructionData(task=self.task,scene=scene,messages=save_history)
        return ins
 
    def generate_question(
        self, chart_data: ChartData, code_data: CodeData, messages: list[Message], scene: str
    ) -> str:
        """基于对话历史生成后续问题的示例"""
        history = "\n".join([f"{msg['role']}: {msg["content"]}" for msg in messages])
        sys_setting = """
        你是一位对图表展示有需求的前端工程师,善于通过不断地提出图表的编辑需求来优化图表效果。
        """
        prompt = PromptTemplate.from_template(
            """
        {sys}
        根据最近的对话历史：
        {history}
        请生成一个与图表编辑相关的后续问题，要求：
        1. 基于前文提到的修改需求,保持问题自然连贯；
        2. 沉浸代入用户视角，模仿用户的口吻，以{language}语言为主要语言，提出的问题尽可能简洁清晰；
        3. 提出的图表需求描述没有必要使用礼貌用词，而是应该尽可能接近真实人类的语言风格，避免使用过多的修饰词和复杂的句式。 
        4. 图表编辑的需求限定场景为{scene}，要求增加或删除或修改
        5. 返回一个字符串，代表你提出的问题。
        """
        )
        prompt_value = prompt.format(sys=sys_setting, history=history,scene=scene ,language=self.language)
        response = self.model.invoke(prompt_value)
        return response.content
