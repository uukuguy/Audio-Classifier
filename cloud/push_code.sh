#!/usr/bin/env bash
# 本机运行: rsync 推送云端必需代码(cloud/ + tools/climb/cycle_context.py)到 AutoDL。
# 本仓库无 git remote, 走 rsync 不走 git clone。
#
# 用法(本机, repo 根):
#   bash cloud/push_code.sh                              # 默认推到 /root/audio-classifier
#   REMOTE_DIR=/root/work bash cloud/push_code.sh        # 自定义目标
set -euo pipefail

SSH_HOST="${SSH_HOST:-root@connect.westd.seetacloud.com}"
SSH_PORT="${SSH_PORT:-46379}"
REMOTE_DIR="${REMOTE_DIR:-/root/audio-classifier}"

echo "[push] target: $SSH_HOST:$REMOTE_DIR  (port $SSH_PORT)"
ssh -p "$SSH_PORT" "$SSH_HOST" "mkdir -p $REMOTE_DIR/{cloud,tools/climb}"

# cloud/ 全推
rsync -avz -e "ssh -p $SSH_PORT" \
  --exclude='__pycache__' \
  cloud/ "$SSH_HOST:$REMOTE_DIR/cloud/"

# tools/climb/cycle_context.py (train_head_cuda 的唯一依赖)
rsync -avz -e "ssh -p $SSH_PORT" \
  tools/climb/cycle_context.py "$SSH_HOST:$REMOTE_DIR/tools/climb/"

echo "[push] done. 云端 ls:"
ssh -p "$SSH_PORT" "$SSH_HOST" "ls -la $REMOTE_DIR/cloud/ $REMOTE_DIR/tools/climb/"
