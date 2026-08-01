# ====== 构建阶段：安装 Python 依赖 ======
FROM python:3.10-slim AS builder

# 配置国内镜像源加速 apt（Debian Trixie 使用 .sources 文件）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources

# 安装编译依赖（chromadb / sentence-transformers 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖清单，利用 Docker 层缓存
COPY requirements.txt .

# 将依赖安装到独立目录，便于下一阶段复制（使用阿里云 pip 镜像加速）
# --mount=type=cache 持久化 pip 下载缓存，改 requirements.txt 后已下载的包不会重新下载
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip install --user -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com


# ====== 运行阶段：精简运行镜像 ======
FROM python:3.10-slim

# 配置国内镜像源加速 apt
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources

# 运行时仍需要 libgomp（sentence-transformers 依赖）和 sqlite3
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 阶段复制已安装的 Python 包
COPY --from=builder /root/.local /root/.local

# 确保 user site 在 PATH 中
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 复制应用代码与语料
COPY backend.py .
COPY app.py .
COPY bm.md .

# 模型与向量库通过 volume 挂载，不打入镜像
# 部署时需挂载:
#   - ./models:/app/models
#   - ./chroma_db:/app/chroma_db
#   - .env 通过 --env-file 注入
VOLUME ["/app/models", "/app/chroma_db"]

# 暴露 Flask 端口
EXPOSE 5000

# 使用 gunicorn 生产级 WSGI 服务器启动
# --workers 2      : 2 个工作进程
# --threads 4      : 每进程 4 线程（支持 SSE 流式并发）
# --timeout 300    : RAG 生成较慢，延长超时
# --bind 0.0.0.0:5000 : 监听所有网卡
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "300", "--bind", "0.0.0.0:5000", "app:app"]
