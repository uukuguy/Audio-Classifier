"""VAP 表征高维可分性探针 (用户洞察: 线性可分弱≠高维不可分).

假设: VAP 冻结表征对 BC 线性可分性弱(r<0.04/LGBM到顶), 但若信号非线性纠缠,
核映射(RBF/随机傅里叶RFF)或 MLP 到高维可能找到超平面。

做法 (云端 CUDA 提冻结 VAP 256d 表征, sklearn 跑分类器对比):
  提取: VAP 冻结 → out["x"] 末窗 mean-pool 256d (不微调, 纯表征)
  对比 BC OOF F1/AUC:
    L. 线性 LogisticRegression (基线: 线性可分性)
    K. RBF-SVM (核映射高维)
    R. 随机傅里叶 RFF → 线性 (高维近似 RBF, 可扩展)
    M. MLP (非线性, 学高维表示)
    T. LGBM (树, 轴对齐非线性, 对照已知)
  若 K/R/M >> L 且 > LGBM → 信号在高维可分, 被线性方法埋没 → VAP 值得.
  若都 ≈ → 表征里 BC 信号确实弱(非线性也救不了).

纯表征(冻结)+ context 可选. 40-80通快验概念.
Usage: python cloud/probe_vap_kernel.py --convs 80 [--with-ctx]
"""
import argparse
import glob
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
torch.set_num_threads(8)  # 限线程防20核爆(云端无限制torch会狂开)

VAP_ROOT = os.environ.get("VAP_ROOT", str(Path(__file__).parent.parent / "baselines/VAP"))
sys.path.insert(0, VAP_ROOT)
sys.path.insert(0, "tools/climb")

CTX, TGT, CHUNK_MS, SR16, SEED = 375, 25, 80, 16000, 42
BC = 2
VAP_CKPT = os.environ.get("VAP_CKPT", str(Path(VAP_ROOT) / "example/checkpoints/VAP_state_dict.pt"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIN_SEC = int(os.environ.get("WIN_SEC", "10"))
POOL = 50


def build_vap():
    from vap.modules.encoder import EncoderCPC
    from vap.modules.modules import TransformerStereo
    from vap.modules.VAP import VAP
    enc = EncoderCPC(load_pretrained=False)
    m = VAP(enc, TransformerStereo(dim=enc.dim))
    sd = torch.load(VAP_CKPT, map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=False)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(DEV).eval()


def load_wav(cid):
    with wave.open(f"data/train/audio/{cid}.wav", "rb") as wf:
        sr = wf.getframerate(); raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0, sr


def augment_wav(s16, rng):
    """音频增强 (增多样性, 生成"不同音频同标签"的BC正例). s16: [2, samples] tensor.
    组合: 加高斯噪声 + gain扰动 + 时间掩码(SpecAug时域近似). 不变速(保因果时序)."""
    x = s16.clone()
    # 1. 加噪 (SNR ~20-30dB)
    noise = torch.randn_like(x) * (x.std() * rng.uniform(0.03, 0.10))
    x = x + noise
    # 2. gain 扰动 (±3dB)
    x = x * float(rng.uniform(0.7, 1.4))
    # 3. 时间掩码 (随机置零一小段, SpecAug 时域版)
    if x.shape[1] > 2000:
        ml = int(rng.uniform(0.02, 0.08) * x.shape[1])
        st = int(rng.uniform(0, x.shape[1] - ml))
        x[:, st:st + ml] = 0.0
    return x


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=80)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--with-ctx", action="store_true")
    args = ap.parse_args()

    from cycle_context import featurize as ctxfeat
    m = build_vap()
    conv = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))[:args.convs]
    print(f"[kernel] {len(conv)} convs dev={DEV} extracting frozen VAP repr...", file=sys.stderr)

    arng = np.random.default_rng(123)
    n_aug = int(os.environ.get("N_AUG", "3"))  # 每个BC正例生成几个增强版
    feats, ctxs, y, G, is_aug = [], [], [], [], []
    for gi, cid in enumerate(conv):
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        wav, sr = load_wav(cid)
        for e in range(CTX, len(a) - TGT + 1, args.stride):
            end8 = int(e * CHUNK_MS / 1000 * sr)
            seg = wav[:, max(0, end8 - WIN_SEC * sr):end8]
            if seg.shape[1] < sr:
                continue
            s16 = torchaudio.functional.resample(torch.from_numpy(seg), sr, SR16)
            need = WIN_SEC * SR16
            s16 = torch.nn.functional.pad(s16, (need - s16.shape[1], 0)) if s16.shape[1] < need else s16[:, -need:]
            ctx = ctxfeat(a[e - CTX:e])
            fut = set(int(x) for x in a[e:e + TGT])
            lab = 1 if BC in fut else 0
            # 原始样本
            out = m(s16.unsqueeze(0).to(DEV))
            feats.append(out["x"][:, -POOL:, :].mean(1).squeeze(0).cpu().numpy())
            ctxs.append(ctx); y.append(lab); G.append(gi); is_aug.append(0)
            # BC 正例增强 (生成 n_aug 个增强版, 标 is_aug=1, OOF 只放 train)
            if lab == 1 and n_aug > 0:
                for _ in range(n_aug):
                    aug = augment_wav(s16, arng)
                    o2 = m(aug.unsqueeze(0).to(DEV))
                    feats.append(o2["x"][:, -POOL:, :].mean(1).squeeze(0).cpu().numpy())
                    ctxs.append(ctx); y.append(1); G.append(gi); is_aug.append(1)
        if (gi + 1) % 20 == 0:
            print(f"[aug] {gi+1}/{len(conv)} convs, {len(feats)} samples ({sum(is_aug)} aug)", file=sys.stderr)

    Xv = np.array(feats, dtype=np.float32)
    y = np.array(y); G = np.array(G); is_aug = np.array(is_aug)
    print(f"[aug] 原始{int((is_aug==0).sum())} + 增强{int(is_aug.sum())} BC正例样本", file=sys.stderr)
    if args.with_ctx:
        Xv = np.concatenate([np.array(ctxs, dtype=np.float32), Xv], axis=1)
    print(f"[kernel] X={Xv.shape} BC率={y.mean():.3f}", file=sys.stderr)

    # 标准化 (核方法必需)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score, roc_auc_score
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(SEED); perm = rng.permutation(len(conv))
    # 评估只在原始样本(is_aug==0); 增强样本只进 train (防 val 虚高)
    orig_mask = (is_aug == 0)
    def oof(make_clf, use_proba=True):
        o = np.full(len(Xv), np.nan)
        for fi in range(5):
            val = {perm[i] for i in range(len(conv)) if i % 5 == fi}
            # train: 本折外所有(含增强); val: 本折内仅原始
            tr = [i for i in range(len(Xv)) if G[i] not in val]
            va = [i for i in range(len(Xv)) if G[i] in val and is_aug[i] == 0]
            sc = StandardScaler().fit(Xv[tr])
            Xtr, Xva = sc.transform(Xv[tr]), sc.transform(Xv[va])
            clf = make_clf()
            clf.fit(Xtr, y[tr])
            o[va] = clf.predict_proba(Xva)[:, 1] if use_proba else clf.decision_function(Xva)
        return o

    from sklearn.pipeline import make_pipeline
    spw = (len(y) - y.sum()) / max(1, y.sum())
    g = 1.0 / Xv.shape[1]
    # 全部快方法 (去掉 SVC probability=True 这个20核狂烧的瓶颈).
    # RFF→线性 = RBF核的高维近似(等价于核SVM但快), 多个 n_components 看维度效应.
    models = {
        "L_linear": lambda: LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
        "R_rff500": lambda: make_pipeline(
            RBFSampler(gamma=g, n_components=500, random_state=SEED),
            LogisticRegression(class_weight="balanced", max_iter=2000)),
        "R_rff2000": lambda: make_pipeline(
            RBFSampler(gamma=g, n_components=2000, random_state=SEED),
            LogisticRegression(class_weight="balanced", max_iter=2000)),
        "M_mlp": lambda: MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                                       early_stopping=True, random_state=SEED),
        "T_lgbm": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                         scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED),
    }
    print(f"\n=== BC 可分性: 线性 vs 核/高维 (VAP冻结表征{'+ctx' if args.with_ctx else ''}) ===")
    res = {}
    ev = orig_mask  # 只在原始样本上评估 (增强样本 o=nan)
    yo = y[ev]
    for name, mk in models.items():
        try:
            o = oof(mk, use_proba=True)[ev]
            auc = roc_auc_score(yo, o)
            bf = max(f1_score(yo, (o >= t).astype(int), zero_division=0)
                     for t in np.linspace(0.05, 0.95, 19))
            res[name] = {"bc_f1": round(float(bf), 4), "auc": round(float(auc), 4)}
            print(f"  {name:<12} BC_F1={bf:.4f}  AUC={auc:.4f}")
        except Exception as ex:
            print(f"  {name:<12} ERR {ex}", file=sys.stderr)
    print("\n判读: 若 K/R/M 的 AUC/F1 >> L 且 > T_lgbm → BC信号在高维可分被线性埋没(VAP值得)")
    print("       若都≈ → 表征BC信号确实弱, 非线性也救不了")
    print(json.dumps({"cycle": "VAP-kernel-probe", "with_ctx": args.with_ctx, "results": res}))


if __name__ == "__main__":
    main()
