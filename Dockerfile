# 使用官方Python基础镜像（直接指定所需Python版本）
FROM python:3.13.2-slim

WORKDIR /app

# 先复制依赖列表（利用Docker缓存层）
COPY requirements.txt .
RUN pip install uv -i https://mirrors.aliyun.com/pypi/simple && uv venv

# 安装依赖（使用清华镜像加速）
RUN uv pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple \
    -r requirements.txt

# 复制项目代码
COPY . .

RUN uv pip list