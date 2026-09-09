# fabric-algo 运行时镜像（纯 TRT 推理 + 模型管理，无 paddle）
# ------------------------------------------------------------------
# 前提：宿主机已装 NVIDIA 驱动 + Docker + nvidia-container-toolkit（见 容器部署.md）
# 构建本机需能访问 pip 源（含 PyTorch CUDA 轮子）。
# engine 不打进镜像（绑 GPU 架构），首次启动/上传模型时在容器内构建。
#
# 构建（在 fabric-algo 目录）:
#   docker build -f deploy/Dockerfile -t fabric-algo:latest .
# 运行推荐用 compose（自动挂载 models/runs/config 与缓存卷）:
#   docker compose -f deploy/docker-compose.yml up -d

FROM nvidia/cuda:12.4.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# python3.12 为 ubuntu24.04 自带；ffmpeg 供 opencv 视频解码兜底
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip python3.12-dev \
        ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python

WORKDIR /app

# 先装依赖（利用层缓存，代码改动不触发重装）
COPY deploy/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

# 代码与静态资源（models/runs 由卷挂载，不打进镜像）
COPY tools/ /app/tools/
COPY static/ /app/static/
COPY docs/ /app/docs/
COPY *.py /app/

# 默认配置用容器路径（compose 会再挂载同名文件以便持久化版本更新）
COPY deploy/config.container.yaml /app/tools/config.yaml

EXPOSE 8001
CMD ["python", "web_server.py", "--host", "0.0.0.0", "--port", "8001"]
