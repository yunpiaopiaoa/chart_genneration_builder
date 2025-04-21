import base64
import configparser
from io import BytesIO
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from src.eval.critic.criteria import Criteria


def get_image(img_path: str):
    """
    获取图片
    :param img_path: 图片路径
    :return: 图片对象
    """
    from PIL import Image

    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return image_data


if __name__ == "__main__":
    # 读取配置
    cur_dir = Path(__file__).resolve().parent
    config_path = cur_dir / "config" / "config.ini"
    con = configparser.ConfigParser()
    con.read(config_path, encoding="utf-8")
    config = con["judge_llm"]
    judge_llm = ChatOpenAI(
        base_url=config["base_url"],
        model=config["model"],
        api_key=config["api_key"],
    )

    # ans=judge_llm.invoke([{'role':'system','content':'你是一个评估模型，给定一张图片和一段文字描述，判断这段文字描述是否符合图片内容。'},
    #                       {'role':'user','content':f"data:image/jpeg;base64,{get_image('output/0/chart.png')}"},
    #                     ])
    # print(ans)
    # image_test=['output/0/chart.png',
    #             'output/3/chart.png',
    #             'output/4/chart.png',
    #             'output/19/chart.png']
    image_test = ["tmp/img/1743955717.png", "tmp/img/bad1.jpg", "tmp/img/bad2.png","tmp/img/bar_num_5.png"]
    crit = Criteria()
    ans = {}
    for image in image_test:
        ans[image] = {"style": {}, "layout": {}}
        for k, c in crit.style.items():
            v = judge_llm.invoke(
                [
                    {"role": "system", "content": c},
                    {"role": "user", "content": "<image>"},
                    {"role": "assistant", "content": "5"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{get_image(image)}"
                                },
                            }
                        ],
                    },
                ]
            )
            ans[image]["style"][k] = v.content
        for k, c in crit.layout.items():
            v = judge_llm.invoke(
                [
                    {"role": "system", "content": c},
                    {"role": "user", "content": "<image>"},
                    {"role": "assistant", "content": "5"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{get_image(image)}"
                                },
                            }
                        ],
                    },
                ]
            )
            ans[image]["layout"][k] = v.content
    with open("tmp/ans_4.json", "w", encoding="utf-8") as f:
        json.dump(ans, f, ensure_ascii=False, indent=4)
