import os,sys,numpy as np
sys.path.insert(0,os.getcwd()); os.environ["MODELS"]=os.getcwd()+"/models"; os.environ["TEST_ROOT"]="/tmp/diag10"
from src.sources.ssl_encoder import infer_ssl
from pathlib import Path
wsp=infer_ssl("whisper",Path("models/wsp_head")); print("现场 wsp:",wsp.mean(0).round(3),flush=True)
hub=infer_ssl("hubert",Path("models/hub_head")); print("现场 hub:",hub.mean(0).round(3),flush=True)
# 对照: 缓存前10段
z=np.load("tools/runs/climb/orthofuse-3src-20260601-1607/fused_probs.npz")
print("缓存 wsp:",z["whisper_te"][:10].mean(0).round(3))
print("缓存 hub:",z["hubert_te"][:10].mean(0).round(3))
