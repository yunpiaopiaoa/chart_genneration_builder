from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Iterable
import json, re
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from numpy import str_
from src.datamodel.infer_result import InferResult, TaskResult
from src.eval.eval_template import EvalTemplateDict
from src.utils.extract import extract_block
from src.utils.img_similarity import img_similarity
from src.utils.dict_similarity import dict_similarity
from src.datamodel.annotation import encode_base64


def replace_content(
    mes: str, replace_dict: dict, infer_query: list[dict] = None
) -> list[dict]:
    parts = re.findall(r"<[^>]*>|\{[^}]*\}|[^<>{}]+", mes)
    result = []
    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            # <image>
            key: str = part[1:-1]
            content_ = replace_dict[key]
            if type(content_) == list and type(content_[0]) == str:
                result.append({"type": "text", "text": "\n".join(content_)})
            elif type(content_) == dict:
                str_json = json.dumps(content_, ensure_ascii=False)
                str_json = str_json.replace("{", "{{")
                str_json = str_json.replace("}", "}}")
                result.append(
                    {
                        "type": "text",
                        "text": str_json,
                        # "text":str(content_)
                    }
                )
            elif type(content_) == str and key.startswith("image"):
                result.append(
                    {"type": "image_url", "image_url": {"url": encode_base64(content_)}}
                )
            elif type(content_) == str:
                result.append({"type": "text", "text": content_})
        elif part.startswith("{") and part.endswith("}"):
            key = part[1:-1]
            # {image}
            if key.startswith("image"):
                result.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": "{" + key + "}"},
                    }
                )
            elif key == "query" and infer_query is not None:
                for q in infer_query:
                    result.append(q)
        else:
            result.append({"type": "text", "text": part})
    return result


def get_query_from_infer(infer: InferResult, task_name: str) -> list[dict]:
    tasks = infer["task_results"]
    question = None
    for task in tasks:
        if task["task"] == task_name:
            question = task["question"]
    infer_query = []
    for q in question:
        infer_query.append({"type": "text", "text": f"\n###{q['role']}:\n"})
        for c in q["content"]:
            infer_query.extend(replace_content(c["value"], infer))
    return infer_query


class EvalTemplate(Runnable):

    def __init__(self, eval_critic_dict: dict, infer_query: list[dict] = None):
        super().__init__()
        messages = []
        self.eval_critic_dict = eval_critic_dict
        for mes in eval_critic_dict["messages"]:
            content = self.formalize_message(mes["content"], infer_query)
            if mes["role"] == "user":
                messages.append(HumanMessagePromptTemplate.from_template(content))
            elif mes["role"] == "system":
                messages.append(SystemMessage(content))
            elif mes["role"] == "assistant":
                messages.append(AIMessage(content))
        self.template = ChatPromptTemplate.from_messages(messages)

    def invoke(self, infer: dict, config: dict = None):
        # print(self.template.messages,file=open('log.txt','w',encoding='utf-8'))
        # print("\n===============\n",file=open('log.txt','a',encoding='utf-8'))
        # print(self.template.invoke(infer),file=open('log.txt','a',encoding='utf-8'))
        return self.template.invoke(infer)

    def formalize_message(self, mes: str, infer_query=None) -> list:
        # parts = re.findall(r'[^<]+|<[^>]+>', mes)
        result = replace_content(mes, self.eval_critic_dict, infer_query)
        return result


class EvalProcess:
    def __init__(
        self, judge_llm: ChatOpenAI, critic_dir: str
    ):
        self.judge_llm = judge_llm
        critic_path = Path(critic_dir)
        self.eval_critcs: dict[str, str] = {}
        for c in critic_path.glob("*.json"):
            with open(c, "r", encoding="utf-8") as f:
                critic_dict = json.load(f)
            self.eval_critcs[c.stem] = critic_dict

    def eval(self, infer_results_dir: Path, tasks: list[str]):
        # eval_results: dict[str, dict[str, list[float]]] = defaultdict(dict)
        infer_res_path = infer_results_dir.joinpath("infer_result.json")
        with open(infer_res_path, "r", encoding="utf-8") as f:
            infer_results:InferResult = json.load(f)
        eval_res = {}
        for task in tasks:
            task_eval_res = {}
            image_path = infer_results_dir.joinpath(task + ".png")
            if not image_path.exists():
                continue
            infer_query = get_query_from_infer(infer_results, task)
            for _, v in self.eval_critcs.items():
                eval_template = EvalTemplate(v, infer_query)
                ans = self.judge_llm.invoke(
                    eval_template.invoke(
                        {
                            "image": encode_base64(image_path),
                        }
                    )
                ).content
                ans = extract_block(ans)
                ans = json.loads(ans)
                for k, v in ans.items():
                    if k != "reason":
                        task_eval_res[k] = v
            eval_res[task] = task_eval_res
        return eval_res
