from copy import deepcopy
import json
import os
from pathlib import Path
from PIL import Image

from src.datamodel.annotation import Annotation


class EvalSample:
    def __init__(self, annotation: Annotation):
        self.annotation = annotation

    @property
    def chart_data(self):
        return self.annotation["chart"]

    @property
    def img_path(self):
        return self.annotation["img_path"]

    def generate_task(self):
        """需要根据annotation实时构造query，以约束模型生成的内容，方便后续评估
        返回值:任务名称,查询问题，真实答案,图片(根据query决定是否需要返回图片，图片后续将作为待评估模型的输入)
        """
        for ins in self.annotation["instructions"]:
            task_name = ins["task"]
            ground_truth = ins["conversations"][-1]
            messages = ins["conversations"][:-1]
            eval_messages = deepcopy(messages)
            # WARNING:这里根据task_name将query中的占位符替换为真实值
            # ground_truth占位符未替换为真实值
            for message in eval_messages:
                for content in message["contents"]:
                    v = content["value"]
                    if content["modality"] == "text":
                        content["value"] = (
                            v.replace(
                                "<chart_data>",
                                json.dumps(self.annotation["chart"]["data"]),
                            )
                            .replace("<code>", self.annotation["code"]["code"])
                            .replace(
                                "<description>", self.annotation["chart"]["description"]
                            )
                        )
                    elif content["modality"] == "image":
                        # 此处把content["value"]=<image>替换为图片对象，方便后续传给大模型调用answer
                        content["value"] = Image.open(self.annotation["img_path"])
            yield task_name, messages, eval_messages, ground_truth


class EvalDataset:
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        for dir in self.path.iterdir():
            if not dir.is_dir():
                continue
            idx = dir.stem
            json_path = dir / "annotation.json"
            with json_path.open("r", encoding="utf-8") as f:
                annotation_data: Annotation = json.load(f)
            yield EvalSample(annotation_data)
