# Next-Session Handoff

**Updated:** 2026-05-29 15:46
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **LoRA 冒烟正在云端跑**（~15:38 启动，预计 ~21min 完成 5 folds）。VRAM 从 OOM(46.4GB) → 1.5GB，所有修复一次到位。
2. **冒烟完成后 = 查 log 确认结果 → 若 BC 有信号 → 跑 lora-full**。
3. **SOTA 不变 = 变体 F 0.7124**。

## LoRA 冒烟状态

**已修问题（本次 commit b2e53d8）**:
- gradient checkpointing：VRAM 从 46.4GB → 1.5GB（-330x）
- gradient accumulation：micro_batch=8 × accum=2 = effective 16
- fold 间 `.cpu()` + `empty_cache()` 释放显存
- predict_oof 死代码清理
- VRAM 监控每 epoch 打印

**冒烟参数**: 40 通, cap5(200 样本), 10ep, 5fold, batch=8, accum=2
**冒烟观察**: fold1 loss 0.73→0.59 收敛, fold2 进行中, ~4.3min/fold

## Next steps（立即）

1. **查冒烟结果**: `ssh -p 46379 root@connect.westd.seetacloud.com 'tail -50 /tmp/lora-smoke.log'`
   - 看 cap1_macro_f1 和 BC F1（frozen baseline BC=0.200, 纯 ctx LGBM=0.222）
   - BC > 0.25 = 有信号值得全量；BC ≤ 0.22 = LoRA 也没救 BC
2. **冒烟有信号 → 跑全量**:
   ```bash
   ssh -p 46379 root@connect.westd.seetacloud.com 'cd /root/audio-classifier && export PATH=/root/miniconda3/bin:$PATH && nohup bash cloud/run_cloud.sh lora-full > /tmp/lora-full.log 2>&1 & echo $!'
   ```
   - lora-full: 全 369 通, cap5(~1845 样本), 50ep, batch=8, accum=4 (effective=32)
   - 预计 ~5-6h（每 fold ~1h × 5 folds）
3. **全量出 CSV → 提交 → 贴分**
4. **冒烟无信号 → 决策门**: ①LoRA 换 target_modules（加 k_proj/o_proj）②换更大 r ③守 0.7124 等复赛

## 云端环境

- AutoDL 4090 在线，SSH: `ssh -p 46379 root@connect.westd.seetacloud.com`
- peft 0.19.1 已装，gradient checkpointing 确认生效
- 代码已推（train_lora.py + run_cloud.sh, commit b2e53d8）
- 冒烟 log: `/tmp/lora-smoke.log`，全量 log 将写 `/tmp/lora-full.log`

## climb 假设池

- **H-L1** (ranking 0.85): LoRA r=32, cap5 切片 — **当前在跑**
- H-L2 (ranking 0.60): LoRA + ASL 损失
- H-L3 (ranking 0.40): LoRA + cap1 (退路)

## Ready commands

```bash
# 查冒烟结果
ssh -p 46379 root@connect.westd.seetacloud.com 'tail -30 /tmp/lora-smoke.log'
# 查全量进度
ssh -p 46379 root@connect.westd.seetacloud.com 'tail -20 /tmp/lora-full.log'
# 推代码（如需再改）
rsync -avz -e "ssh -p 46379" cloud/ root@connect.westd.seetacloud.com:/root/audio-classifier/cloud/
```
