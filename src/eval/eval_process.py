from collections import defaultdict
import json
import numbers
import os
from pathlib import Path
from typing import Iterable


from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_core.messages import BaseMessage

from src.datamodel.infer_result import InferResult, TaskResult
from src.eval.eval_template import EvalTemplateDict
from src.utils.extract import extract_block
from src.utils.img_similarity import img_similarity
from src.utils.dict_similarity import dict_similarity


class EvalProcess:
    def __init__(self, judge_llm: ChatOpenAI):
        self.judge_llm = judge_llm
        self.eval_templates = EvalTemplateDict()
        vision_chains: dict[str, Runnable] = {}
        for critic_name, template in self.eval_templates.vision_templates():
            vision_chains[critic_name] = template | self.judge_llm
        self.vision_chain = RunnableParallel(**vision_chains)

        text_chains: dict[str, Runnable] = {}
        for critic_name, template in self.eval_templates.text_templates():
            text_chains[critic_name] = template | self.judge_llm
        self.text_chain = RunnableParallel(**text_chains)

    def eval(self, infer_results: Iterable[tuple[InferResult, Path]]):
        eval_results: dict[str, dict[str, list[float]]] = defaultdict(dict)
        for infer_result, infer_path in infer_results:
            for task_result in infer_result["task_results"]:
                task_name = task_result["task"]
                if task_name.endswith("type"):  # img2type,code2type
                    metrics = self._eval_type(
                        task_result["prediction"],
                        task_result["ground_truth"]["content"][0]["value"],
                    )
                elif task_name.endswith("code"):  # img2code,data2code,text2code
                    code = extract_block(task_result["prediction"])  # 提取代码块
                    metrics = self._eval_code(
                        code,
                        infer_result["img_path"],
                        infer_path / f"{task_name}.png",
                        task_name == "img2code",
                    )
                elif task_name.endswith("data"):  # img2data,code2data,text2data
                    data_block = extract_block(task_result["prediction"])
                    #WARNING:如果为字典列表，而不是key:[value]格式，需要转换格式
                    try:
                        pred_dict = json.loads(data_block)
                        if isinstance(pred_dict, list):
                            convert_dict = {}
                            for dic in pred_dict:
                                for k, v in dic.items():
                                    convert_dict.setdefault(k, []).append(v)
                            pred_dict = convert_dict
                        metrics = self._eval_data(
                            pred_dict, infer_result["chart_data"]["data"]
                        )
                    except json.JSONDecodeError as e:  # 如果json解析失败
                        print(f"Error: {e}")
                        metrics = {"data_similarity": 0.0}
                elif task_name.endswith("text"):  # img2text
                    metrics = self._eval_text(task_result, infer_result["img_path"])
                elif task_name == "qa":
                    metrics = self.eval_qa(
                        task_result,
                        infer_result["chart_data"]["data"],
                        infer_result["img_path"],
                        infer_result["code"]["code"]
                    )
                else:
                    raise ValueError(f"Unsupported task: {task_name}")
                for item, score in metrics.items():
                    eval_results[task_name].setdefault(item, []).append(score)
                    # print(f"{task_name} {item}: {score}")
        return eval_results

    def eval_qa(
        self, task_result: TaskResult, chart_data: dict[str, list], img_path: str,code:str
    ):
        """评估qa问答与原图表的相关性"""
        prompt_value = self.eval_templates.qa_template().invoke(
            {
                "chart_data": chart_data,
                "code": code,
                "img_path": img_path,
                "query": task_result["question"][-1],  ##WARNING:严重的耦合
                ##task_result["question"]是待评估模型需要回答的query，包含图表数据，代码和图片,用户询问
                ##task_result["question"][-1]是用户询问
                "answer": task_result["prediction"],
            }
        )
        response = self.judge_llm.invoke(prompt_value)
        return {"relation": int(response.content)}

    def _eval_type(self, predicted_type: str, true_type: str):
        """比较预测图表类型和真实图表类型"""
        metrics = {"type_similarity": int(predicted_type == true_type)}
        return metrics

    def _eval_data(self, pred_dict: TaskResult, chart_data: dict[str, list]):
        # print(pred_dict)
        score = dict_similarity(pred_dict, chart_data)
        return {"data_similarity": score}

    def _eval_text(self, task_result: TaskResult, img_path: str):
        metrics: dict[str, float] = {}
        responses: dict[str, BaseMessage] = self.text_chain.invoke(
            {"img_path": img_path, "description": task_result["prediction"]}
        )
        for critic_name, response in responses.items():
            score = int(response.content)
            metrics[critic_name] = score
        return metrics

    def _eval_code(
        self,
        code: str,
        img_path: str,
        infer_img_path: str,
        need_img_similarity: bool,
    ):
        metrics: dict[str, float] = {}
        try:
            assert os.path.exists(infer_img_path)
            metrics["code_success_rate"] = 1
            if need_img_similarity:
                metrics["img_similarity"] = float(
                    img_similarity(img_path, infer_img_path)
                )
            # 计算图片视觉评价
            responses: dict[str, BaseMessage] = self.vision_chain.invoke(infer_img_path)
            for critic_name, response in responses.items():
                content = response.content
                if content.isalnum():
                    score = int(response.content)
                else:
                    try:
                        data = json.loads(content)
                        ##content附带的评分理由会被过滤掉
                        scores = [
                            v for v in data.values() if isinstance(v, numbers.Number)
                        ]
                        score = sum(scores) / len(scores)
                    except Exception as e:
                        print(e)
                metrics[critic_name] = score
        except Exception as e:
            print(f"Error: {e}")
            metrics["code_success_rate"] = 0
            if need_img_similarity:
                metrics["img_similarity"] = 0.0
            for critic_name, template in self.eval_templates.vision_templates():
                metrics[critic_name] = 0.0
        return metrics
