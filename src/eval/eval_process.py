from collections import defaultdict
from curses.ascii import isalnum
import json
import numbers
from pathlib import Path


from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_core.messages import BaseMessage

from src.datamodel.infer_result import InferResult, TaskResult
from src.eval.eval_template import EvalTemplateDict
from src.build.generator.img_gen.base_img_generator import BaseImgGenerator
from src.utils.img_similarity import img_similarity
from src.utils.dict_similarity import dict_similarity


class EvalProcess:
    def __init__(self, judge_llm: ChatOpenAI, chart_img_gen: BaseImgGenerator):
        self.judge_llm = judge_llm
        self.chart_img_gen = chart_img_gen
        self.eval_templates = EvalTemplateDict()
        vision_chains: dict[str, Runnable] = {}
        for critic_name, template in self.eval_templates.vision_templates():
            vision_chains[critic_name] = template | self.judge_llm
        self.vision_chain = RunnableParallel(**vision_chains)

        text_chains: dict[str, Runnable] = {}
        for critic_name, template in self.eval_templates.text_templates():
            text_chains[critic_name] = template | self.judge_llm
        self.text_chain = RunnableParallel(**text_chains)

    def eval(self, infer_results: list[InferResult], eval_path: Path):
        eval_results: dict[str, dict[str, list[float]]] = defaultdict(dict)
        for infer_result in infer_results:
            for task_result in infer_result["task_results"]:
                task_name = task_result["task"]
                if task_name.endswith("type"):  # img2type,code2type
                    metrics = self._eval_type(
                        task_result["prediction"],
                        task_result["ground_truth"]["contents"][0]["value"],
                    )
                elif task_name.endswith("code"):  # img2code,data2code,text2code
                    metrics = self._eval_code(
                        task_result["prediction"],
                        infer_result["img_path"],
                        task_name,
                        task_name == "img2code",
                        eval_path,
                    )
                elif task_name.endswith("data"):  # img2data,code2data,text2data
                    metrics = self._eval_data(
                        task_result, infer_result["chart_data"]["data"]
                    )
                elif task_name.endswith("text"):  # img2text
                    metrics = self._eval_text(task_result, infer_result["img_path"])
                elif task_name == "qa":
                    metrics = self.eval_qa(task_result)
                else:
                    raise ValueError(f"Unsupported task: {task_name}")
                for item, score in metrics.items():
                    eval_results[task_name].setdefault(item, []).append(score)
                    print(f"{task_name} {item}: {score}")
        return eval_results

    def eval_qa(self, task_result: TaskResult):
        """评估qa问答与原图的相关性
        TODO
        """
        return {"relation": 1}

    def _eval_type(self, predicted_type: str, true_type: str):
        """比较预测图表类型和真实图表类型
        TODO:后续可能允许真实类型和预测类型相近即可，需要定义图表类型之间的相似分数矩阵
        """
        metrics = {"type_similarity": int(predicted_type == true_type)}
        return metrics

    def _eval_data(self, task_result: TaskResult, chart_data: dict[str, list]):
        score = dict_similarity(json.loads(task_result["prediction"]), chart_data)
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
        task_name: str,
        need_img_similarity: bool,
        eval_path: Path,
    ):
        metrics: dict[str, float] = {}
        try:
            eval_img_path = str(
                eval_path
                / "img"
                / f"{"_".join(img_path.split("/")[:2])}_{task_name}.png"
            )
            self.chart_img_gen.generate_img(code, eval_img_path)
            metrics["code_success_rate"] = 1
            if need_img_similarity:
                metrics["img_similarity"] = float(img_similarity(img_path, eval_img_path))
            # 计算图片视觉评价
            responses: dict[str, BaseMessage] = self.vision_chain.invoke(eval_img_path)
            for critic_name, response in responses.items():
                content=response.content
                print(critic_name,content)
                if content.isalnum():
                    score = int(response.content)
                else:
                    try:
                        data=json.loads(content)
                        ##content附带的评分理由会被过滤掉
                        scores=[v for v in data.values() if isinstance(v, numbers.Number)]
                        score=sum(scores)/len(scores)
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
