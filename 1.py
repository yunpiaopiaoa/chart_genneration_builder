import configparser
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.datamodel.annotation import encode_base64

cur_dir = Path(__file__).resolve().parent

# 读取配置
config_dir = cur_dir / "config"
config_path = config_dir / "config.ini"
con = configparser.ConfigParser()
con.read(config_path, encoding="utf-8")
llm = ChatOpenAI(
    base_url=con["build_llm"]["base_url"],
    model=con["build_llm"]["model"],
    api_key=con["build_llm"]["api_key"],
)


image1='sample/sample2/0/chart.png'
image2='sample/sample2/10/chart.png'
template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(""),
        HumanMessage(
            [
                {
                    "type": "text",
                    "text": "这是图片1，",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_base64(image1)
                    },
                },
                {
                    "type": "text",
                    "text": "这是图片2，",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_base64(image2)
                    },
                },
                {
                    "type": "text",
                    "text": "请问图片1和图片2分别是什么类型的图片，他们有什么区别？，那个更符合美学原则？",
                },
            ]
        ),])
print(template.invoke({}))
# ans=llm.invoke(template.invoke({}))
# print(ans.content)