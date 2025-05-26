FROM python:3.13.2-slim
WORKDIR /app

# 安装uv并创建虚拟环境
RUN pip install uv -i https://mirrors.huaweicloud.com/repository/pypi/simple && \
    uv venv /app/.venv

# 确保后续命令使用虚拟环境
ENV PATH="/app/.venv/bin:$PATH" \
    BASH_ENV="/app/.venv/bin/activate" 
RUN echo 'source /app/.venv/bin/activate' >> ~/.bashrc

# 安装依赖
COPY requirements.txt .
RUN uv pip install --no-cache-dir \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple \
    -r requirements.txt

# 复制代码
COPY . .