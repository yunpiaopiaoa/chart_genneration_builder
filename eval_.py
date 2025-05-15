from collections import defaultdict
import json,re
import numbers
import os
from pathlib import Path
from pyexpat.errors import messages
from typing import Iterable
import configparser

from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_core.messages import BaseMessage

from src.datamodel.infer_result import InferResult, TaskResult
from src.eval.eval_template import EvalTemplateDict
from src.utils.extract import extract_block
from src.utils.img_similarity import img_similarity
from src.utils.dict_similarity import dict_similarity
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.datamodel.annotation import encode_base64

class EvalTemplate(Runnable):
    
    def __init__(self,eval_critic_dict:dict,infer_query:list[dict]=None):
        super().__init__()
        messages=[]
        self.eval_critic_dict=eval_critic_dict
        for mes in eval_critic_dict['messages']:
            content=self.formalize_message(mes['content'])
            if mes['role']=='user':
                messages.append(HumanMessagePromptTemplate.from_template(content,infer_query))
            elif mes['role']=='system':
                messages.append(SystemMessage(content,infer_query))
            elif mes['role']=='assistant':
                messages.append(AIMessage(content,infer_query))
        self.template=ChatPromptTemplate.from_messages(messages)
    
    def invoke(self, infer: dict, config: dict=None):
        return self.template.invoke(infer)
    
    def formalize_message(self,mes:str,query=None)->list:
        # parts = re.findall(r'[^<]+|<[^>]+>', mes)
        parts = re.findall(r'<[^>]*>|\{[^}]*\}|[^<>{}]+', mes)
        result = []
        for part in parts:
            if part.startswith("<") and part.endswith(">"):
                # <image>
                key:str=part[1:-1]
                content_=self.eval_critic_dict[key]
                if type(content_)==list:
                    result.append({
                        "type": "text",
                        "text": "\n".join(content_)
                    })
                elif type(content_)==dict:
                    result.append({
                        "type": "text",
                        "text": json.dumps(content_, ensure_ascii=False)
                    })
                elif type(content_)==str and key.startswith("image"):
                    result.append({
                        "type": "image_url",
                        "image_url": {"url": encode_base64(content_)}
                    })
                elif type(content_)==str:
                    result.append({
                        "type": "text",
                        "text": content_
                    })
            elif part.startswith("{") and part.endswith("}"):
                key=part[1:-1]
                # {image}
                if key.startswith("image"):
                    result.append({
                        "type": "image_url",
                        "image_url": {"url": "{"+key+"}"},
                    })
                elif key=='query':
                    for mes in query:
                        result.append({
                            "type": "text",
                            "text": f'\n###{mes['role']}:\n'
                        })
                        for con in mes['content']:
                            if con['type']=='text':
                                result.append({
                                    "type": "text",
                                    "text": con['text']
                                })
                            elif con['type']=='image':
                                result.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": con['image_url']['url']
                                    }
                                })
                            
            
            else:
                result.append({
                    "type": "text",
                    "text": part
                })
        return result
        

    

class EvalProcess:
    def __init__(self, judge_llm: ChatOpenAI,critic_dir: str = "src/eval/critic2/vision"):
        self.judge_llm = judge_llm
        critic_path=Path(critic_dir)
        self.eval_dimensions:dict[str,EvalTemplate]={}
        for c in critic_path.glob("*.json"):
            with open(c, "r", encoding="utf-8") as f:
                critic_dict = json.load(f)
            self.eval_dimensions[c.stem] = EvalTemplate(critic_dict)                                                 

# def get_json(text: str)->json:
#     st=text.find("{")
#     end=text.rfind("}")
#     if st == -1 or end == -1:
#         print("No JSON found in the text.\n",text)
#         return None
#     return json.loads(text[st:end + 1])

if __name__ == "__main__":
    
    con = configparser.ConfigParser()
    con.read('config/config.ini', encoding="utf-8")
    config = con["judge_llm"]
    judge_llm = ChatOpenAI(
        base_url=config["base_url"],
        model=config["model"],
        api_key=config["api_key"],
        temperature=0,##确保评估稳定性
    )
    ep = EvalProcess(judge_llm,critic_dir="src/eval/critic2/vision")
    print(ep.eval_dimensions)
    images_dir_path=Path('manual_annotations/images')
    annotations_dir_path=Path('test/eval_pingfen')
    os.makedirs(annotations_dir_path,exist_ok=True)
    for image_path in images_dir_path.glob("*.png"):
        image_name=image_path.stem
        annotation_path=annotations_dir_path/f"{image_name}.json"
        annotathon_={}
        for _,v in ep.eval_dimensions.items():
            ans= judge_llm.invoke(v.invoke(image_path)).content
            ans=extract_block(ans)
            for k,v in ans.items():
                if k!='reason':
                    annotathon_[k]=v
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(annotathon_, f, ensure_ascii=False, indent=4)
        
    
    