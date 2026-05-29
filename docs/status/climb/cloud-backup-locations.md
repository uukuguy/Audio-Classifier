# 云端备份索引 (2026-05-28 关机前)

云主机: ssh -p 46379 root@connect.westd.seetacloud.com (已建互信)

## /root/autodl-fs/backups/ (200G 长期盘, 关机/释放都不丢)
| 路径 | 内容 | 大小 |
|---|---|---|
| whisper_cache_full/ | 369 train + 1000 test 帧特征 npz + _done 标记 | 64G |
| manual_models/whisper-large-v3/ | safetensors 全套, 不用重下 | 2.9G |
| cloud-whisper-full/ | 全量 3 份 CSV + cv_metrics.json | 69K |
| cloud-whisper-smoke/ | 冒烟产物 | 20K |
| .done | backup 完成 sentinel (内容 BACKUP_DONE) | 12B |

## /root/audio-classifier/ (系统盘 127G, 关机 60 天内不丢)
- 全部代码 + 64G whisper_cache + ~10G conda env

## 下次开机恢复 (代码已 push 过, 不需再传)
```bash
# 1. 软链或拷贝 cache (推荐链接节省盘)
ln -sf /root/autodl-fs/backups/whisper_cache_full /root/audio-classifier/data/whisper_cache
ln -sf /root/autodl-fs/backups/manual_models /root/.cache/manual_models

# 2. 直接跑 head 即可 (跳过提取)
cd /root/audio-classifier
RUN_DIR=tools/runs/climb/<新run> bash cloud/run_cloud.sh head
```

## 当前 paradigm 真分等待中
- pred_test1.csv (cycle1 阈值, 默认推荐提交) → 等用户提交公榜
- 真分回来用 `bash tools/climb/apply-lb-score.sh "cloud-whisper-full <分数>"` 注入

## 三大 cap1 OOF 候选 macro (cap1 OOF 上的, 非线上预测)
- cap1 阈值: 0.6521 (BC@0.80 极激进, 不建议提交)
- cycle1 阈值: 0.6250 (BC@0.50 阈值铁律最稳, 默认提交)
- balanced 阈值: 0.6320 (BC@0.40 略激进, 备选)
