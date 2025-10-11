import io, json, sys
from pathlib import Path
from typing import List
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from torchvision import transforms
import sys 
import torch

REPO = Path("/home/sagemaker-user/src/LostPetTest")
ROOT_DIR = Path("/home/sagemaker-user/src/Mine/dog").resolve()
MODEL_PATH = REPO / "output_petface" / "best_model.pth"
INDEX_PATH = REPO / "output_petface" / "gallery_index_small.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(str(REPO))
from model.make_model import make_model
from config_petface import cfg



app = FastAPI(title="PetFace Demo API")
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="/home/sagemaker-user/src/LostPetTest/frontend/dist", html=True), name="app")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
app.mount("/images", StaticFiles(directory=str(ROOT_DIR)), name="images")

transform = transforms.Compose([
    transforms.Resize(cfg.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD),
])

model = make_model(
    backbone_name=cfg.BACKBONE, 
    num_classes=400,
    embed_dim=cfg.EMBED_DIM,
    pretrained=False  # 推理阶段不需要下载预训练权重
)

ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)

possible_keys = ["state_dict", "model", "net", "weights"]
state = None
if isinstance(ckpt, dict):
    for k in possible_keys:
        if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            print(f"=> using ckpt['{k}'] as state_dict (len={len(state)})")
            break
    # 如果顶层本身就是 state_dict（键都是权重名），也接受
    if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
        print(f"=> using top-level ckpt as state_dict (len={len(state)})")

if state is None:
    raise RuntimeError(f"Cannot find state_dict in checkpoint. Top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

missing, unexpected = model.load_state_dict(state, strict=False)
print("=> missing keys (first 5):", missing[:5], " ... total", len(missing))
print("=> unexpected keys (first 5):", unexpected[:5], " ... total", len(unexpected))

model.to(DEVICE).eval()

# --- 读取索引：兼容两种 paths 存法 ---
file = np.load(str(INDEX_PATH), allow_pickle=True)
gallery_embeds = file["embeddings"].astype(np.float32)  # [N, D], 已归一化

paths_obj = file["paths"]
if hasattr(paths_obj, "item"):  # numpy scalar 包的 json 字符串
    gallery_paths: List[str] = json.loads(paths_obj.item())
else:
    # 也可能直接是 np.array(list_of_str)
    gallery_paths = list(paths_obj.tolist())

def extract_feature(pil_img: Image.Image):
    with torch.inference_mode():
        t = transform(pil_img).unsqueeze(0).to(DEVICE)
        feat = model(t)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

@app.post("/api/search")
async def search(image: UploadFile = File(...), top_k: int = Form(5)):
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    q = extract_feature(pil)                 # (D,)
    sims = gallery_embeds @ q                # 余弦（已归一化）
    k = max(1, min(int(top_k), len(sims)))
    idx = np.argsort(-sims)[:k]
    results = []
    for i in idx.tolist():
        rel = gallery_paths[i]
        results.append({
            "path": rel,
            "url": f"/images/{rel}",
            "score": float(sims[i]),
            "label": rel.split("/")[-2:],
        })
    return {"results": results}