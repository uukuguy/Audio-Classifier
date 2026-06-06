# FinVCup 2026 复赛推理镜像 — ctx-only 骨架 (T4 step1)
#
# 设计:
#   - 单 ENTRYPOINT 跑 src.infer, 任意长度 context 适配 (T1 D-26)
#   - 推理纯 CPU (LGBM), 不依赖 GPU. R4 全栈升级后再换 CUDA base
#   - 镜像目标 < 500MB (python:3.12-slim + lgbm/sklearn + ~5MB ckpt)
#
# 用法 (本机验证):
#   docker build -t finvcup-infer:ctx-only .
#   docker run --rm \
#       -v $PWD/data/test:/data/test:ro \
#       -v $PWD/tools/runs/climb/_docker_out:/output \
#       finvcup-infer:ctx-only
#
# 镜像约定 (跟 baseline run_infer.sh 风格对齐, 复赛操作手册公布前的预设):
#   /data/test/    ← test_root, 含 context/<id>.npy (+ audio/text 后续升级用)
#   /output/       ← 写 pred_test1.csv 出来
#   /app/models/   ← ckpt (镜像构建期 COPY 进来, 不接受外部上传)

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4

# 系统依赖 (LGBM 需 libgomp1, sklearn 需 libstdc++)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖 (镜像 < 500MB 的关键; 不装 transformers/torch, ctx-only 用不到)
COPY requirements.docker.txt /app/requirements.docker.txt
RUN pip install --no-cache-dir -r /app/requirements.docker.txt

# 代码 (src/ 单入口 + tools/climb 复用的 featurize / normalize_ctx_to_375)
COPY src/ /app/src/
COPY tools/climb/cycle_context.py /app/tools/climb/cycle_context.py
COPY tools/climb/dynamic_ctx_utils.py /app/tools/climb/dynamic_ctx_utils.py
COPY tools/climb/__init__.py /app/tools/climb/__init__.py
RUN touch /app/tools/__init__.py

# 模型 ckpt (镜像构建期固化, 不来自外部上传 = joblib.load 信任边界)
COPY models/ctx_only/ /app/models/ctx_only/

# 输出目录 mount point
RUN mkdir -p /output

# 默认入口: --test_root 和 --output_csv 走默认值, 可被 `docker run` 覆盖
ENTRYPOINT ["python", "-m", "src.infer", \
            "--ckpt_dir", "/app/models/ctx_only", \
            "--test_root", "/data/test", \
            "--output_csv", "/output/pred_test1.csv"]
