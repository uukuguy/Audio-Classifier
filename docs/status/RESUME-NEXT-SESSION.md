# Next-Session Handoff

**Updated:** 2026-05-29 16:43
**恢复命令：`/project-state resume`**（lightweight-memory，非 gsd）

## TL;DR

1. **LoRA 冒烟成功：BC=0.267（+20% vs LGBM 0.222），信号确认**。
2. **LoRA 全量训练正在云端跑**（PID=8857，~16:43 启动，预计 ~21:30 完成 5 folds）。
3. **冒烟 CSV 已拉回本机**（`tools/runs/climb/lora-smoke/pred_test1.csv`），可先提交看线上分。
4. **SOTA 不变 = 变体 F 0.7124**。

## Currently Running

| 任务 | PID | 启动 | 预计完成 | 查进度 |
|---|---|---|---|---|
| LoRA 全量训练 (369通/50ep/5fold) | 云端 8857 | 16:43 | ~21:30 | `ssh -p 46379 root@connect.westd.seetacloud.com 'tail -20 /tmp/lora-full.log'` |

## Next steps（按优先级）

1. **提交冒烟 CSV 看线上分**：`tools/runs/climb/lora-smoke/pred_test1.csv`（40通欠拟合，预期低于 SOTA 但验证 LoRA 线上表现）
2. **等全量完成**（~21:30）→ 查 log → 拉 CSV → 提交 → 贴分
3. **全量有信号 → 进一步优化**：①LoRA + ASL 损失 ②加 k_proj/o_proj target_modules ③增大 r
4. **全量无信号 → 决策门**：守 0.7124 等复赛

## 冒烟详细结果

- cap1 CV macro=0.5901 | C=0.987 T=0.435 **BC=0.267** I=0.400 NA=0.862
- BC 突破确认（frozen=0.200, ctx LGBM=0.222）
- T/I 比 LGBM 差（40 通欠拟合），全量应补回

## 云端环境

- AutoDL 4090 在线，SSH: `ssh -p 46379 root@connect.westd.seetacloud.com`
- 全量 log: `/tmp/lora-full.log`，checkpoint: `tools/runs/climb/lora-full/fold{N}.pt`
- 每 fold 完成自动保存 checkpoint（修复后）

## Ready commands

```bash
# 查全量进度
ssh -p 46379 root@connect.westd.seetacloud.com 'tail -20 /tmp/lora-full.log'
# 查是否完成
ssh -p 46379 root@connect.westd.seetacloud.com 'kill -0 8857 2>/dev/null && echo RUNNING || echo DONE'
# 拉全量 CSV
scp -P 46379 root@connect.westd.seetacloud.com:/root/audio-classifier/tools/runs/climb/lora-full/pred_test1.csv tools/runs/climb/lora-full/
# 推代码
rsync -avz -e "ssh -p 46379" cloud/ root@connect.westd.seetacloud.com:/root/audio-classifier/cloud/
```

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
