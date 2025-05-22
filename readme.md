# 图表生成与评测系统
## 项目概述
本项目是一个完整的图表生成、推理和评测系统，包含三个主要阶段：
1. 图表生成阶段 - 生成图表数据、代码和图片和指令响应对
2. 推理阶段 - 待评测模型根据前一阶段生成的指令进行推理
3. 评测阶段 - 评估推理结果的质量

其中评测的任务有10类:
+ code2data (代码到数据)
+ data2code (数据到代码)
+ data2type (数据到类型)
+ img2code (图片到代码)
+ img2data (图片到数据)
+ img2text (图片到描述)
+ img2type (图片到类型)
+ text2code (文本到代码)
+ text2data (文本到数据)
+ 问答任务(QA)

每一类任务下有若干个评测指标。
+ x2data类型：
    - 评测推理所得图表数据与原始图表数据的相似性（通过字典相似度的规则算法）
+ x2code类型：
    - 代码通过率（是否可渲染为图片）
    - 视觉指标（使用大模型判断）：
        - 颜色
        - 字体
        - 布局
        - 线宽
+ x2type类型：
    - 直接比较推理结果与原始图表类型，或者通过图表类型相似矩阵判断
+ img2text任务：
    - 使用大模型判断文本描述的准确性
+ QA任务：
    - ...
## 快速开始
### 运行步骤
1. 创建环境
```
conda env create -f environment.yml
```
2. 配置config/config.ini

将config/config.ini.template更名为config/config.ini

对于build_llm，eval_llm,judge_llm，填写具体的模型访问参数;

3. 下载echarts.min.js到lib目录下
```
wget -P lib https://cdn.bootcss.com/echarts/4.9.0/echarts.min.js 
```

4. 生成图表数据
需配置build_llm参数
```
python build.py --sample_dir=sample --gen_count=2
```
其中sample_dir是生成结果的目录，gen_count是生成图表数据样本的数量

5. 执行推理任务
需配置eval_llm参数
```
python infer_multithread3.py --sample_dir=sample --infer_dir=results_infer
```
其中sample_dir是图表数据样本目录，应与build.py的sample_dir参数一致

infer_dir是推理结果的目录

6. 评估推理结果
需配置judge_llm参数
```
python eval_with_pandas.py --infer_dir=results_infer --eval_dir=results_eval
```
其中infer_dir是推理结果目录，应与infer.py的infer_dir参数一致
评测过程对于代码评测任务，代码渲染图片将保存至tmp/img目录下
## 目录结构
```
.
├── config/                 # 配置文件
|   └── config.ini          # 配置文件
├── lib/                    # 第三方库，如存放echarts.min.js
├── src/
│   ├── builld/        
|   |   ├── code_gen    # 代码生成器
|   |   ├── data_gen    # 图表数据生成器
|   |   ├── img_gen     # 图片生成器
|   |   └── instruction_gen # 指令生成器 
│   ├── datamodel/          # 数据模型定义
|   |   ├── annotation.py   # 样本数据定义（包含图表数据、代码对象、指令数据）
|   |   ├── infer_result.py # 推理结果定义
|   |   └── task_type.py    # 评估任务类型集合
│   ├── infer/              # 推理相关代码
|   |    ├── evaldataset.py # 评测样本数据集
|   |    ├── eval_gpt.py    # 待评测GPT模型（子类）
|   |    └── toeval_llm.py  # 待评测大模型（基类）
│   └── eval/               # 评测相关代码
|       ├── critic/         # 存放评测指标的评分标准json文件
|       ├── eval_process/   # 针对推理结果分任务获取评测分数
|       └── template/       # 大模型评测对话模板
├── build.py                # 图表生成脚本  
├── infer.py                # 推理脚本
└── eval.py                 # 评测脚本   
```
## 系统架构
### 图表生成阶段

功能:
+ 生成图表数据 (JSON格式)
+ 生成ECharts/Python代码
+ 渲染图表图片
+ 生成各种任务的指令数据

生成流程:

1. 使用ChartxDataGenerator生成图表数据
2. 使用EchartsHtmlGeneratorLLM生成ECharts代码
3. 使用EchartsImgGenerator渲染图表图片
4. 使用InstructionGen生成各种任务的指令

### 推理阶段
功能:针对每个评测样本，待评估模型需面向多种图表相关任务给出推理结果。


### 评测阶段

## 扩展性
系统设计具有良好的扩展性，可以轻松添加：
+ 新的生成器实现（图表样本、代码、图片生成）
    - 在src/build/generator的子目录下基于Base类添加新的生成器子类
+ 新的评测任务类型
    - 在src/datamodel/task_type.py中定义新的评测任务类型字符串
    - 实现一个BaseInstructionTemplate子类，并且在InstructionGen类init函数中添加模板
+ 新的评测标准
    - 根据需要构造对话模板
    - 在src/eval/critic目录下添加新的评测标准json文件
    - 在EvalTemplateDict中读取评测标准json文件，并在EvalProcess类的init函数中构造可以执行invoke的链（多个评测标准的对话实例可以同时向大模型输入）


### 注意事项
build.py构造出来的指令中，会使用如下占位符：

`<chart_data>`代表图表数据

`<code>`代表代码字符串

`<description>`代表图表描述

`<image>`代码图表图像

使用占位符的目的是降低样本的存储空间

在infer阶段，评测样本会调用generate_task函数，返回值包括messages, eval_messages
+ messages仍保留占位符，方便推理结果存储
+ eval_messages中的占位符将被替换为真实数据或者图像，输入到待评测模型中进行推理。