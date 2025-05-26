import base64
from io import BytesIO
import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from src.datamodel.annotation import Message
from src.infer.toeval_llm import BaseToEvalLLM


class EvalGpt(BaseToEvalLLM):
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    # handler = logging.FileHandler(f"log/{__name__}.log")
    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    def __init__(self, model: ChatOpenAI):
        super().__init__()
        self.model = model

    def answer(self, messages: list[Message]):
        input = []
        for message in messages:
            if isinstance(message["content"], str):
                input.append(message)
            elif isinstance(message["content"], list):
                contents = []
                for content in message["content"]:
                    if content["type"] == "text":
                        contents.append({"type": "text", "text": content["value"]})
                    elif content["type"] == "image":
                        contents.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": content["value"]
                                },
                            }
                        )
                    else:
                        raise ValueError(f"Unsupported type {content["type"]}")
                dic = {"role": message["role"], "content": contents}
                input.append(dic)
            else:
                raise ValueError(f"Unsupported content type {type(message['content'])}")
        # self.logger.info(str(messages))
        response = self.model.invoke(input)
        return response.content
