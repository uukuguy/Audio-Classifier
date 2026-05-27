# FinVCup 2026 — 上云提取/训练 + 复赛推理镜像（单一真相）
#
# 设计:thin layer over 官方 pytorch CUDA 镜像，torch 2.7.1 对齐本机(MPS↔CUDA 一处 device 切换)。
# AutoDL 用法:控制台选 PyTorch 2.7.1 / CUDA 12.1 base 镜像，再 pip install -r requirements.txt
#            （AutoDL 公共零售不支持 push 任意 Dockerfile，故以本文件为"规格"，pip freeze 对齐）。
# RunPod/阿里云/腾讯云:可直接 docker build + push 此镜像 → 与复赛提交镜像 1:1 一致。
#
# 复赛镜像:此文件即提交镜像基底。推理入口 src.infer_test（CUDA），加 ENTRYPOINT 即可。

FROM pytorch/pytorch:2.7.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf \
    PYTHONUNBUFFERED=1

# ffmpeg 给 torchaudio 解码兜底（wav 用不到，复赛若有其它格式备用）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# 代码在运行时挂载/上传到 /workspace（避免每次改码重 build）
# 复赛最终镜像:把 src/ + 模型权重 COPY 进来并设 ENTRYPOINT

CMD ["bash"]
