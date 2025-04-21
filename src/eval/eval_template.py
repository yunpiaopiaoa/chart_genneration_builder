import base64
from io import BytesIO
import json
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
                                "url": f"data:image/jpeg;base64,{encode_base64(case_input_path)}"
                            },
                        },
                    ]
                ),
                AIMessage(
                    [
                        {
                            "type": "text",
                            # "text": case_output,
                            "text": json.dumps(case_output,ensure_ascii=False),
                        },
                    ],
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                        },
                    ]
                ),
            ]
        )

    def invoke(self, img_path: str, config: dict):
        return self.template.invoke({"image_data": encode_base64(img_path)})


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
                                "url": f"data:image/jpeg;base64,{encode_base64(case_img_path)}"
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
                            "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
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
                "image_data": encode_base64(input["img_path"]),
                "description": input["description"],
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

    def vision_templates(self):
        return self._vision_dic.items()

    def text_templates(self):
        return self._text_dic.items()
