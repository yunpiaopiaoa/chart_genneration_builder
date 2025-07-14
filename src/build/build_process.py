from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from tqdm import tqdm
from src.build.code_gen.echarts_html_generator_llm import EchartsHtmlGeneratorLLM
from src.build.data_gen.data_generator_llm import LLMDataGenerator
from src.build.img_gen.echarts_img_generator import EchartsImgGenerator
from src.build.instruction_gen.instruction_gen import InstructionGen
from src.datamodel.annotation import Annotation


class BuildProcess:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(f"log/{__name__}.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    def __init__(
        self,
        llm,
        language: str,
    ):
        self.llm = llm
        # 定义生成器
        # data_gen = ChartxDataGenerator(**dict("../datasets/ChartX"))
        self.data_gen = LLMDataGenerator(llm, language)
        self.code_gen = EchartsHtmlGeneratorLLM(llm)
        self.chart_img_gen = EchartsImgGenerator()
        self.instruction_gen = InstructionGen(llm, language)

    def __del__(self):
        if self.chart_img_gen:
            self.chart_img_gen.cleanup()
            self.chart_img_gen = None

    def helper(self, executor: ThreadPoolExecutor, limited_types, sample_dir: Path, i):
        try:
            # 第一阶段
            future_chart_data = executor.submit(
                self.data_gen.generate_data, limited_types
            )
            chart_data = future_chart_data.result()
            chart_type = chart_data["type"]
            target_dir = sample_dir / chart_type / f"{i}"
            target_dir.mkdir(parents=True, exist_ok=True)
            # 第二阶段：生成代码
            future_code = executor.submit(self.code_gen.generate_code, chart_data)
            # 第三阶段：写HTML + 生成图片 + 生成指令
            future_code.add_done_callback(
                lambda future=future_code, target_dir=target_dir: (
                    target_dir / "index.html"
                ).write_text(future.result()["code"], encoding="utf-8")
            )
            future_code.add_done_callback(
                lambda future=future_code, target_dir=target_dir: self.chart_img_gen.generate_img(
                    future.result()["code"],
                    target_dir / "chart.png",
                )
            )
            future_code.add_done_callback(
                lambda future=future_code, chart_data=chart_data, target_dir=target_dir: self.gen_annotation(
                    chart_data,
                    future.result(),
                    target_dir,
                    target_dir / "chart.png",
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to build at {i}, error: {e}")

    def build(self, gen_count: int, sample_dir: Path, limited_types=None):
        sample_dir.mkdir(parents=True, exist_ok=True)
        exist_count = sum(
            item.is_dir()
            for sub_dir in sample_dir.iterdir()
            for item in sub_dir.iterdir()
        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in tqdm(range(exist_count, exist_count + gen_count)):
                self.helper(executor, limited_types, sample_dir, i)

    def gen_annotation(self, chart_data, code_data, target_dir: Path, img_path: str):
        instructions = self.instruction_gen.generate_instruction(
            chart_data, code_data, img_path
        )
        annotation = Annotation(
            chart=chart_data,
            code=code_data,
            instructions=instructions,
        )
        (target_dir / "annotation.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=4), encoding="utf-8"
        )
