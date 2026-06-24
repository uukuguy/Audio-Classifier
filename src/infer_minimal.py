"""最简推理 — 仅 Context LGBM, 无 CUDA/transformers 依赖。验证基础管线。"""
import glob, json, sys, os
from pathlib import Path
import numpy as np, joblib

TEST = Path(os.environ.get("TEST_ROOT", "/xydata"))
OUT = Path(os.environ.get("OUTPUT_CSV", "/app/submit/submit.csv"))
MODELS = Path(os.environ.get("MODELS", "/app/models"))
CTX, NA = 375, 4
SUB = ["c","na","i","bc","t"]
C2K = {"c":0,"na":4,"i":3,"bc":2,"t":1}
THR = [0.05, 0.50, 0.75, 0.65, 0.25]

def featurize(ctx):
    NUM=5;oh=np.eye(NUM)[ctx];feats=[]
    for w in (10,25,50,100,200,375):feats.extend(oh[-min(w,len(oh)):].mean(axis=0))
    for i in range(1,6):feats.append(ctx[-i] if len(ctx)>=i else -1)
    L=len(ctx)
    for k in range(NUM):
        pos=np.where(ctx==k)[0];feats.append((L-1-pos[-1])/L if len(pos) else 1.0)
    for k in range(NUM):feats.append((ctx==k).sum()/L)
    feats.append((ctx[1:]!=ctx[:-1]).mean() if L>1 else 0.0)
    return np.array(feats,dtype=np.float32)

def pad_375(ctx):
    n=len(ctx)
    if n>=CTX: return ctx[-CTX:].astype(np.int32)
    return np.concatenate([np.full(CTX-n,NA,dtype=np.int32),ctx.astype(np.int32)])

try:
    ctx_dir = MODELS / "ctx_only"
    clfs = {k: joblib.load(ctx_dir / f"lgbm_{['C','T','BC','I','NA'][k].lower()}.joblib") for k in range(5)}
    thr_json = json.loads((ctx_dir / "thresholds.json").read_text())
    thrs = [thr_json[lab.lower()] for lab in ["C","T","BC","I","NA"]]
    
    files = sorted(glob.glob(str(TEST / "context/*.npy")))
    print(f"Found {len(files)} context files", flush=True)
    if not files:
        print("NO FILES FOUND", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT,"w") as f: f.write("segment_id,"+",".join(SUB)+"\n")
        sys.exit(0)
    
    sids = [Path(p).stem for p in files]
    X = np.array([featurize(pad_375(np.load(p).astype(int))) for p in files])
    probs = np.zeros((len(X),5), dtype=np.float32)
    for k in range(5):
        probs[:,k] = clfs[k].predict_proba(X)[:,1]
    
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT,"w") as f:
        f.write("segment_id,"+",".join(SUB)+"\n")
        for i,sid in enumerate(sids):
            vals = [str(int(probs[i,C2K[c]]>=THR[C2K[c]])) for c in SUB]
            f.write(f"{sid},"+",".join(vals)+"\n")
    
    pos = {c:int((probs[:,C2K[c]]>=THR[C2K[c]]).sum()) for c in SUB}
    print(f"Done. {OUT} pos={pos}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with open(OUT,"w") as f: f.write("segment_id,"+",".join(SUB)+"\n")
    sys.exit(1)
