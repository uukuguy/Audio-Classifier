import os, sys, numpy as np
sys.path.insert(0, os.getcwd())
os.environ["MODELS"]=os.getcwd()+"/models"
os.environ["TEST_ROOT"]="/tmp/diag50"
from src.sources.context import infer_context
from src.sources.ssl_encoder import infer_ssl
from pathlib import Path
ctx,seg = infer_context()
print("现场 ctx mean [C,T,BC,I,NA]:", ctx.mean(0).round(3), flush=True)
wsp = infer_ssl("whisper", Path("models/wsp_head"))
print("现场 wsp mean:", wsp.mean(0).round(3), flush=True)
hub = infer_ssl("hubert", Path("models/hub_head"))
print("现场 hub mean:", hub.mean(0).round(3), flush=True)
np.savez("/tmp/diag_live.npz", ctx=ctx, wsp=wsp, hub=hub, seg=np.array(seg))
print("saved n=", len(seg), flush=True)
