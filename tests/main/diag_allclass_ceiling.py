"""全类天花板诊断: 在选攻击目标前看清战场.
每类的 OOF P/R/F1 现状 + 不同阈值下的可达上限 + macro 分解.
关键问: 是不是真只有 BC 有空间? T/I/NA 边际如何?
本机, 用真值会话级 5fold OOF.
"""
import glob, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "tools/climb")
from cycle_context import featurize
from sklearn.metrics import f1_score, precision_recall_fscore_support
from lightgbm import LGBMClassifier

LAB = {0:"C",1:"T",2:"BC",3:"I",4:"NA"}
NUM, CTX, TGT, SEED = 5, 375, 25, 42
STRIDE = 40

conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
X, Y, G = [], [], []
for gi, cid in enumerate(conv_ids):
    a = np.load(f"data/train/labels/{cid}.npy").astype(int)
    for e in range(CTX, len(a)-TGT+1, STRIDE):
        X.append(featurize(a[e-CTX:e])); 
        fut=set(int(x) for x in a[e:e+TGT])
        Y.append([1 if k in fut else 0 for k in range(NUM)]); G.append(gi)
X,Y,G = np.array(X),np.array(Y),np.array(G)
print(f"samples={len(X)}", file=sys.stderr)

rng=np.random.default_rng(SEED); perm=rng.permutation(len(conv_ids))
oof=np.zeros((len(X),NUM))
for fi in range(5):
    val={perm[i] for i in range(len(conv_ids)) if i%5==fi}
    tr=[i for i in range(len(X)) if G[i] not in val]; va=[i for i in range(len(X)) if G[i] in val]
    for k in range(NUM):
        y=Y[tr,k]; spw=(len(y)-y.sum())/max(1,y.sum())
        c=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,scale_pos_weight=spw,n_jobs=4,verbose=-1,random_state=SEED)
        c.fit(X[tr],y); oof[va,k]=c.predict_proba(X[va])[:,1]

print(f"\n=== 全类 OOF 天花板 (全切片高分辨率) ===")
print(f"{'类':>4}{'正例率':>8}{'最优F1':>8}{'P':>7}{'R':>7}{'thr':>6}{'macro占比':>10}")
f1s={}
for k in range(NUM):
    yt=Y[:,k]; bt,bf=0.5,-1
    for t in np.linspace(0.05,0.95,37):
        f=f1_score(yt,(oof[:,k]>=t).astype(int),zero_division=0)
        if f>bf: bf,bt=f,t
    p,r,_,_=precision_recall_fscore_support(yt,(oof[:,k]>=bt).astype(int),average='binary',zero_division=0)
    f1s[k]=bf
    print(f"{LAB[k]:>4}{yt.mean():>8.3f}{bf:>8.3f}{p:>7.3f}{r:>7.3f}{bt:>6.2f}{bf/5:>10.4f}")
macro=np.mean(list(f1s.values()))
print(f"\nmacro={macro:.4f} (这是滑窗OOF, 线上更高约+0.07)")
print(f"\n=== 边际分析: 每类提升 0.05 对 macro 的贡献 (都是 ÷5) ===")
print("  所有类提升对 macro 贡献相同(等权), 但难度不同:")
for k in range(NUM):
    headroom = 1.0 - f1s[k]  # 理论剩余空间
    print(f"  {LAB[k]}: 当前{f1s[k]:.3f} 剩余空间{headroom:.3f} {'★最弱' if f1s[k]==min(f1s.values()) else ''}")
