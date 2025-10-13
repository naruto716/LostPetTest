import io, json, sys
from pathlib import Path
from typing import List
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from torchvision import transforms

# ==== 路径配置 ====
REPO = Path("/home/sagemaker-user/src/LostPetTest")
ROOT_DIR = Path("/home/sagemaker-user/src/Mine/dog").resolve()
MODEL_PATH = "/home/sagemaker-user/src/LostPetTest/outputs/sweeps_20251011_211632/lr0.0003_m0.3_bnfalse_e20/best_model.pth"
INDEX_PATH = "/home/sagemaker-user/src/LostPetTest/outputs/gallery_index_small.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(str(REPO))
from model.make_model import make_model
from config_petface import cfg

# ==== FastAPI 初始化 ====
app = FastAPI(title="PetFace Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ==== 静态资源 ====
app.mount("/images", StaticFiles(directory=str(ROOT_DIR)), name="images")

# ==== 模型加载 ====
transform = transforms.Compose([
    transforms.Resize(cfg.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD),
])

model = make_model(
    backbone_name="dinov3_vitl16",
    num_classes=0,
    embed_dim=cfg.EMBED_DIM,
    pretrained=False,
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
    if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
        print(f"=> using top-level ckpt as state_dict (len={len(state)})")

if state is None:
    raise RuntimeError(f"Cannot find state_dict in checkpoint. Top-level keys: {list(ckpt.keys())}")

missing, unexpected = model.load_state_dict(state, strict=False)
print("=> missing keys:", len(missing), "unexpected keys:", len(unexpected))

model.to(DEVICE).eval()

# ==== 载入图库索引 ====
file = np.load(str(INDEX_PATH), allow_pickle=True)
gallery_embeds = file["embeddings"].astype(np.float32)

paths_obj = file["paths"]
if hasattr(paths_obj, "item"):
    gallery_paths: List[str] = json.loads(paths_obj.item())
else:
    gallery_paths = list(paths_obj.tolist())

# ==== 特征提取函数 ====
def extract_feature(pil_img: Image.Image):
    with torch.inference_mode():
        t = transform(pil_img).unsqueeze(0).to(DEVICE)
        feat = model(t)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

# ==== 搜索 API ====
@app.post("/api/search")
async def search(image: UploadFile = File(...), top_k: int = Form(5)):
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    q = extract_feature(pil)
    sims = gallery_embeds @ q
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

# ==== 健康检查 ====
@app.get("/healthz")
def health():
    return {"status": "ok"}

# ==== 最后挂载前端静态文件（放在最后！）====
app.mount(
    "/",
    StaticFiles(directory="/home/sagemaker-user/src/LostPetTest/frontend/dist", html=True),
    name="frontend",
)
