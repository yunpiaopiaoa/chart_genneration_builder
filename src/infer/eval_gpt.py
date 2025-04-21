import base64
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from src.datamodel.annotation import Message
from src.infer.toeval_llm import BaseToEvalLLM


class EvalGpt(BaseToEvalLLM):
    def __init__(self, model: ChatOpenAI):
        super().__init__()
        self.model = model

    def answer(self, messages: list[Message]):
        input = []
        for message in messages:
            contents = []
            for content in message["contents"]:
                if content["modality"] == "text":
                    contents.append({"type": "text", "text": content["value"]})
                elif content["modality"] == "image":
                    img = content["value"]
                    buffered = BytesIO()
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(buffered, format="JPEG")
                    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    contents.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        }
                    )
                else:
                    raise ValueError(f"Unsupported modality {content['modality']}")
            dic = {"role": message["role"], "content": contents}
            input.append(dic)
        response = self.model.invoke(input)
        return response.content
