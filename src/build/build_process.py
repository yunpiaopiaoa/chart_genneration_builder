from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from tqdm import tqdm
from src.datamodel.annotation import Annotation
from src.build.generator.code_gen.base_code_generator import BaseCodeGenerator
from src.build.generator.data_gen.base_data_generator import BaseDataGenerator
from src.build.generator.img_gen.base_img_generator import BaseImgGenerator
from src.build.generator.instruction_gen.instruction_gen_llm import InstructionGen


class BuildProcess:
    def __init__(
        self,
        llm,
        data_gen: BaseDataGenerator,
        code_gen: BaseCodeGenerator,
        chart_img_gen: BaseImgGenerator,
        instruction_gen: InstructionGen,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        handler = logging.FileHandler("log/build.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.llm = llm
        self.data_gen = data_gen
        self.code_gen = code_gen
        self.chart_img_gen = chart_img_gen
        self.instruction_gen = instruction_gen

    def build(self, gen_count: int, sample_dir: Path):
        sample_dir.mkdir(parents=True,exist_ok=True)
        exist_count = sum(item.is_dir() for item in sample_dir.iterdir())

        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in tqdm(range(exist_count, exist_count + gen_count)):
                try:
                    target_dir = sample_dir / f"{i}"
                    target_dir.mkdir(exist_ok=True)
                    chart_data = self.data_gen.generate_data()

                    # 第一阶段：保存数据 + 生成代码
                    future_code = executor.submit(
                        self.code_gen.generate_code, chart_data
                    )
                    executor.submit(
                        (target_dir / "data.json").write_text,
                        json.dumps(chart_data, ensure_ascii=False, indent=4),
                        encoding="utf-8",
                    )
                    code_data = future_code.result()

                    # 第二阶段：写HTML + 生成指令 + 生成图片
                    executor.submit(
                        (target_dir / "index.html").write_text,
                        code_data["code"],
                        encoding="utf-8",
                    )
                    executor.submit(
                        self.gen_annotation,
                        chart_data,
                        code_data,
                        target_dir,
                    )

                    # 生成图片（浏览器实例不能跨线程操作）
                    img_path = target_dir / "chart.png"
                    self.chart_img_gen.generate_img(code_data["code"], str(img_path))
                except Exception as e:
                    self.logger.error(f"Failed to build at {i}, error: {e}")
        self.chart_img_gen.cleanup()

    def gen_annotation(self, chart_data, code_data, target_dir: Path):
        instructions = self.instruction_gen.generate_instruction(chart_data, code_data)
        annotation = Annotation(
            chart=chart_data,
            code=code_data,
            img_path=f"{target_dir.parent.stem}/{target_dir.stem}/chart.png",
            instructions=instructions,
        )
        (target_dir / "annotation.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=4), encoding="utf-8"
        )
