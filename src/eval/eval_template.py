import base64
from io import BytesIO
import json
from operator import inv
from pathlib import Path
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from PIL import Image

from src.datamodel.annotation import encode_base64


class EvalImgTemplate(Runnable):
    def __init__(self, critic: str, case_input_path: str, case_output: dict):
        self.template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(critic),
                HumanMessage(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_base64(case_input_path)
                            },
                        },
                    ]
                ),
                AIMessage(
                    [
                        {
                            "type": "text",
                            # "text": case_output,
                            "text": json.dumps(case_output, ensure_ascii=False),
                        },
                    ],
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": "{image_url}"},
                        },
                    ]
                ),
            ]
        )

    def invoke(self, img_path: str, config: dict):
        return self.template.invoke({"image_url": encode_base64(img_path)})


class EvalTextTemplate(Runnable):
    def __init__(
        self, critic: str, case_img_path: str, case_description: str, case_output: str
    ):
        self.template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(critic),
                HumanMessage(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_base64(case_img_path)
                            },
                        },
                        {
                            "type": "text",
                            "text": case_description,
                        },
                    ]
                ),
                AIMessage(
                    [
                        {
                            "type": "text",
                            "text": case_output,
                        },
                    ],
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": "{image_url}"},
                        },
                        {
                            "type": "text",
                            "text": "{description}",
                        },
                    ]
                ),
            ]
        )

    def invoke(self, input: dict, config: dict):
        return self.template.invoke(
            {
                "image_url": encode_base64(input["img_path"]),
                "description": input["description"],
            }
        )


class EvalQATemplate(Runnable):
    def __init__(self):
        critic = [
            "你需要执行qa问答任务评估。评分标准如下：",
            "0分: 完全错误或无关答案",
            "1分: 部分正确但关键信息错误",
            "2分: 主要内容正确但存在细节偏差",
            "3分: 完全正确但表述不精确",
            "4分: 精确匹配且包含补充证据",
            "用户参考了图表数据，代码和图片，对于询问做出了回答。请你对用户的回答给出评分，直接返回一个数字即可。"
        ]
        
        self.template = ChatPromptTemplate.from_messages(
            [
                SystemMessage("\n".join(critic)),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": "图表数据为：{chart_data}",
                        },
                        {
                            "type": "text",
                            "text": "代码为：{code}",
                        },
                        {
                            "type": "text",
                            "text": "图片为：",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "{image_url}"},
                        },
                        {
                            "type": "text",
                            "text": "询问为：{query}",
                        },
                        {
                            "type": "text",
                            "text": "回答为：{answer}",
                        },
                    ]
                ),
            ]
        )
    def invoke(self, input, config = None, **kwargs):
        return self.template.invoke(
            {
                "chart_data": input["chart_data"],
                "code": input["code"],
                "image_url": encode_base64(input["img_path"]),
                "query": input["query"],
                "answer": input["answer"],
            }
        )


class EvalTemplateDict:
    def __init__(self):
        self._vision_dic: dict[str, EvalImgTemplate] = {}
        self._text_dic: dict[str, EvalTextTemplate] = {}
        cur_dir = Path(__file__).parent / "critic1"
        for p in (cur_dir / "vision").glob("*.json"):
            with open(p, "r") as f:
                json_data = json.load(f)
                self._vision_dic[p.stem] = EvalImgTemplate(
                    "\n".join(json_data["critic"]),
                    json_data["case_input_path"],
                    json_data["case_output"],
                )
        for p in (cur_dir / "text").glob("*.json"):
            with open(p, "r") as f:
                json_data = json.load(f)
                self._text_dic[p.stem] = EvalTextTemplate(
                    "\n".join(json_data["critic"]),
                    json_data["case_img_path"],
                    json_data["case_description"],
                    json_data["case_output"],
                )
        self._qa_template = EvalQATemplate()

    def qa_template(self):
        return self._qa_template

    def vision_templates(self):
        return self._vision_dic.items()

    def text_templates(self):
        return self._text_dic.items()
