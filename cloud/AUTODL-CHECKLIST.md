# AutoDL 上云操作单 — whisper-large-v3 攻 BC

> 你做的:注册/充值/选卡/报备 + 贴 SSH 或贴日志回来。其余(环境/代码/提取/训练)我全包。
> 锚:cycle1 线上 0.7108(排22)。可信 CV=cap1 切片 0.656。BC=0.22 是瓶颈。
> 目标:whisper 把 BC 拉到 ~0.40 → 线上 ~0.747 → 进前 3。

---

## 0. 投钱前先想清楚(可信判断已就绪)

- 切片 CV 审计(`docs/status/2026-05-28-sliced-cv-audit.md`)已确认 **BC 是真瓶颈非 CV 假象**——三档采样都钉 0.22。
- 成本极低:全量提取估 **¥70–180**(RTX 4090 @ ¥2/hr)。**这是模型赌注不是钱的赌注。**
- 建议:**先充 ¥50 跑冒烟**(前 40 通,几块钱),看 whisper 帧在 40 通上 BC 有没有抬头;有信号再全量。

---

## 1. 你做:注册 + 实名 + 充值

1. autodl.com 注册,**实名认证**(绑手机/支付宝,国内 GPU 平台强制)
2. 充值 ¥50 起步(支付宝/微信)

## 2. 你做:选卡开机(照此选)

| 项 | 选什么 | 理由 |
|---|---|---|
| **地区** | 任意有 4090 库存的(优先"西北/内蒙"便宜区) | — |
| **GPU** | **RTX 4090 (24GB) × 1** | frozen whisper 提取 fp16 ≤16GB,4090 够且最省;A100 不值 |
| **镜像** | **PyTorch 2.7.1 / CUDA 12.1 / Python 3.12**(选最接近的) | 对齐本机 torch 2.7.1,MPS↔CUDA 一处 device 切换 |
| **数据盘** | ≥ 50GB | whisper 缓存 + train audio |
| **计费** | 按量(冒烟阶段);确认后可包日 | 冒烟便宜 |

开机后:
- 控制台开 **"学术资源加速"**(内置 HF/github 代理,`from_pretrained` 能直连;我的脚本也有 curl 兜底)
- 复制 **SSH 命令**(形如 `ssh -p XXXXX root@region.autodl.com`)备用

## 3. 你做(择一搭桥,我摸不到云实例)

- **方式 A(更稳,推荐)**:我把脚本写好(已就绪),你 ssh 上去 `git clone` 本仓库(或上传 cloud/ 目录),`bash cloud/run_cloud.sh smoke`,把输出贴回来给我看。
- **方式 B**:你把 SSH 连接信息贴给我(注意这等于给我云访问权,你决定)。

## 4. 你做:复赛报备(若用 whisper-large-v3 = 公开模型)

- **2026-06-10 前** 邮件 `xinyebei@xinye.com` 报备使用 whisper-large-v3(openai 公开模型)。
- 报备发件邮箱用**你的比赛注册邮箱**(对外身份,不是系统里的 Claude 账号邮箱)。

---

## 5. 数据上传 (本项目走官方 zip 包, 不打包)

官方提供 `finv11th_train_test_data.zip` (~18GB, 内含 `train/` `test/` 两层)。
你直接上传到云端 `/root/autodl-fs/finv11th_train_test_data.zip`,setup 脚本自动解压套层成 `data/train/...`(脚本零改)。

> 早期 `pack_data.sh` 已废弃——是本机打包再上传的旧路线,改用官方 zip 直传更直接。

---

## 6. 云端跑(本项目实测路径,2026-05-28)

**实测实例**: 4090 48G 显存 / 127G 系统盘 (够,余量 43%) / Python 3.12 镜像 / SSH 互信已建。

```bash
# A. 本机: rsync 推送代码到云 (本仓库无 git remote, 走 rsync 不走 git clone)
bash cloud/push_code.sh                  # 推 cloud/ + tools/climb/cycle_context.py → /root/audio-classifier

# B. 云端: 一键 setup (解压 + 装依赖 + 下模型, 不跑训练)
ssh -p 46379 root@connect.westd.seetacloud.com
cd /root/audio-classifier
bash cloud/setup_cloud.sh                # 跑完打印冒烟命令

# C. 云端: 冒烟 (前 40 通, ~1.2GB 缓存, 验证 BC 是否抬头)
mkdir -p tools/runs/climb/cloud-whisper-smoke
nohup bash cloud/run_cloud.sh smoke > tools/runs/climb/cloud-whisper-smoke/run.log 2>&1 &
echo $! > tools/runs/climb/cloud-whisper-smoke/run.pid
tail -f tools/runs/climb/cloud-whisper-smoke/run.log     # 看到 EXTRACT_COMPLETE + cap1切片CV macro=... + ★BC=... 即冒烟完

# D. 冒烟有信号 → 全量 (~41GB 缓存, 余量 43% 够)
RUN_DIR=tools/runs/climb/cloud-whisper-full nohup bash cloud/run_cloud.sh full \
  > tools/runs/climb/cloud-whisper-full/run.log 2>&1 &

# E. 拿 CSV 回本机 → 手动提交公榜 → 贴回真分 → climb 注入 calibration
scp -P 46379 root@connect.westd.seetacloud.com:/root/audio-classifier/tools/runs/climb/cloud-whisper-smoke/pred_test1.csv .
```

**断点续跑**: spot 被抢/中断, 重跑 `run_cloud.sh extract` 只补未完成的通 (`data/whisper_cache/<split>/_done/` 记进度)。

**判断 BC 信号阈值** (cap1 切片 CV 上):
- BC F1 < 0.25 → 没信号,whisper 也救不动,守 0.7124 别全量
- BC F1 0.25-0.35 → 弱信号,值得全量赌一把
- BC F1 > 0.35 → 强信号,全量必跑

---

## 7. 复赛镜像(我已备 Dockerfile)

`cloud/Dockerfile`(FROM pytorch/pytorch:2.7.1-cuda12.1)= 复赛提交镜像基底,单一真相。
- AutoDL 零售:以此为规格,`pip install -r cloud/requirements.txt` 后存私有镜像,`pip freeze` 对齐。
- 若要提交镜像 1:1 一致 → 改用 RunPod(可直接 docker build+push 此镜像)。

---

## 关键产物清单(我已写好,在 `cloud/`)

| 文件 | 作用 |
|---|---|
| `Dockerfile` / `requirements.txt` | 复赛镜像 + 依赖锚定(torch 2.7.1) |
| `download_whisper.sh` | whisper-large-v3 curl 直下(绕 hf client) |
| `extract_whisper_cuda.py` | CUDA fp16 帧提取,断点续跑,PID+artifact 双信号 |
| `train_head_cuda.py` | 神经头(cross-attn over 帧)+ cap1 切片 CV + 出 CSV |
| `pack_data.sh` | 本机数据打包上传 |
| `run_cloud.sh` | 一键 smoke/full/extract/head |
