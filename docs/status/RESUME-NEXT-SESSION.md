# Next-Session Handoff

**Updated:** 2026-05-29 14:53
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **LoRA 脚本已写好并云端冒烟 3 次，修了 2 个 bug（ctx_dim/dtype），卡在 OOM** — batch_size=16 太大，双声道 encoder 激活撑爆 48GB VRAM。
2. **下一步 = 降 batch_size 到 4-8 重跑冒烟**，然后全量。
3. **SOTA 不变 = 变体 F 0.7124**。

## LoRA 冒烟进展（cycle 11 H-L1）

**已修 bug**:
- `ctx_dim` 从硬编码 80 改为从数据推断（实际 46 维）
- encoder 输出 bf16 → proj 是 fp32，在 proj 前转 float32

**当前阻塞**: OOM on RTX 4090 48GB。根因 = batch_size=16 × 双声道分别过 encoder = 32 次 forward，激活不释放。
- 已分配 46.44 GiB / 47.37 GiB total
- **修法**: `run_cloud.sh` 里 `lora-smoke` 的 `--batch-size 16` 改成 `--batch-size 4`（或直接在 train_lora.py 里加 gradient checkpointing）

## Next steps（立即）

1. **修 `run_cloud.sh` lora-smoke 的 batch-size 为 4**，推送，重跑冒烟
2. 冒烟通过 → `bash cloud/run_cloud.sh lora-full`（batch_size 也可能需降到 8-16）
3. 全量出 CSV → 提交 → 贴分

## 代码状态

| 文件 | commit | 说明 |
|---|---|---|
| `cloud/train_lora.py` | `156eab4` | LoRA 训练脚本（2 bug 已修，OOM 待修） |
| `cloud/run_cloud.sh` | `dfb4903` | 加了 `lora-smoke` / `lora-full` 子命令 |
| `cloud/requirements.txt` | `dfb4903` | 加了 `peft>=0.10.0` |
| `.claude/climb/hypotheses.yaml` | 本地 | 加了 H-L1/H-L2/H-L3（gitignored） |

## 云端环境

- AutoDL 4090 在线，SSH 可达 `root@connect.westd.seetacloud.com:46379`
- peft 已装（0.19.1）
- 代码已推（train_lora.py / run_cloud.sh / requirements.txt）
- `/tmp/lora-smoke.log` 有最后一次 OOM 日志

## Ready commands

```bash
/project-state resume
# SSH 直连云端:
ssh -p 46379 root@connect.westd.seetacloud.com
# 推代码:
bash cloud/push_code.sh
# 修 batch 后重跑冒烟:
rsync -avz -e "ssh -p 46379" cloud/run_cloud.sh root@connect.westd.seetacloud.com:/root/audio-classifier/cloud/
ssh -p 46379 root@connect.westd.seetacloud.com "cd /root/audio-classifier && export PATH=/root/miniconda3/bin:\$PATH && bash cloud/run_cloud.sh lora-smoke"
```

## climb 假设池新增

- **H-L1** (ranking 0.85): LoRA r=32, cap5 切片，cycle 11 主力
- H-L2 (ranking 0.60): LoRA + ASL 损失
- H-L3 (ranking 0.40): LoRA + cap1 (退路)
