import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms

# ========= 配置（可以按需调整） =========
RATIO = 0.10           # 只抽样 10%
MAX_IMAGES = 3000      # 最多处理这么多张，进一步控内存/时间
EMBED_DTYPE = np.float16  # 用 float16 存盘（体积减半），服务端再转 float32

# === 路径设置 ===
REPO = Path("/home/sagemaker-user/src/LostPetTest")
ROOT_DIR = Path("/home/sagemaker-user/src/Mine/dog").resolve()
MODEL_PATH = REPO / "output_petface" / "best_model.pth"
INDEX_PATH = REPO / "output_petface" / "gallery_index_small.npz"  # 新文件名
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 模块导入 ===
import sys
sys.path.append(str(REPO))
from model.make_model import make_model
from config_petface import cfg

# === 图像预处理 ===
transform = transforms.Compose([
    transforms.Resize(cfg.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD),
])

# === 构建模型并加载权重 ===
model = make_model(
    backbone_name=cfg.BACKBONE,
    num_classes=400,
    embed_dim=cfg.EMBED_DIM,   # 你训练日志显示最终 feat_dim=768
    pretrained=False
)

# ✅ 加载权重（适配 ckpt 格式）
ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
state = None
if isinstance(ckpt, dict):
    for k in ("state_dict", "model", "net", "weights"):
        if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]; break
    if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
if state is None:
    raise RuntimeError(f"Cannot find state_dict in checkpoint. Top keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
missing, unexpected = model.load_state_dict(state, strict=False)
print("=> missing:", len(missing), " unexpected:", len(unexpected))
model.to(DEVICE).eval()

# === 准备图片列表（抽样 + 上限） ===
print(f"Scanning images in {ROOT_DIR} ...")
all_images = list(ROOT_DIR.rglob("*.jpg")) + list(ROOT_DIR.rglob("*.png"))
if len(all_images) == 0:
    raise SystemExit("⚠️ 没找到图片，请检查 ROOT_DIR")

random.seed(42)
random.shuffle(all_images)

# 先按比例取，再截断到上限
target_n = max(1, int(len(all_images) * RATIO))
target_n = min(target_n, MAX_IMAGES)
subset_images = all_images[:target_n]
print(f"Selected {len(subset_images)} / {len(all_images)} images (~{int(RATIO*100)}%, cap {MAX_IMAGES})")

# === 先跑一张确定特征维度（避免拼接带来的峰值内存） ===
def extract_feat(pil_img):
    with torch.inference_mode():
        t = transform(pil_img).unsqueeze(0).to(DEVICE)
        feat = model(t)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.squeeze(0).detach().cpu().numpy()

# 试第一张有效图片，确定 D
tmp_feat = None
tmp_idx = None
for i, p in enumerate(subset_images):
    try:
        pil = Image.open(p).convert("RGB")
        tmp_feat = extract_feat(pil)
        tmp_idx = i
        break
    except Exception as e:
        print("skip (probe):", p, e)

if tmp_feat is None:
    raise SystemExit("⚠️ 抽样集合里没有可读的图片")

D = tmp_feat.shape[0]
N = len(subset_images)
# 预分配：使用 float16，后续服务端可以 .astype(np.float32)
embeddings = np.zeros((N, D), dtype=EMBED_DTYPE)
paths = [None] * N

# 先写入探测到的那一张，避免重复算
embeddings[0] = tmp_feat.astype(EMBED_DTYPE)
paths[0] = str(subset_images[tmp_idx].relative_to(ROOT_DIR))

# === 正式提取 ===
print(f"Extracting features: N={N}, D={D}, dtype={embeddings.dtype}")
pbar_iter = tqdm(range(N), desc="Embedding")
for i in pbar_iter:
    if i == 0:
        continue  # 已写
    img_path = subset_images[i]
    try:
        pil = Image.open(img_path).convert("RGB")
        feat = extract_feat(pil).astype(EMBED_DTYPE)
        embeddings[i] = feat
        paths[i] = str(img_path.relative_to(ROOT_DIR))
    except Exception as e:
        # 出错时，留空为 0 向量，同时路径填 None
        print("skip:", img_path, e)
        embeddings[i] = 0
        paths[i] = None

# 过滤掉失败样本（None 路径）
valid_mask = np.array([p is not None for p in paths], dtype=bool)
embeddings = embeddings[valid_mask]
paths = [p for p in paths if p is not None]

if embeddings.shape[0] == 0:
    raise SystemExit("⚠️ 没有成功处理的图片，退出")

# === 存盘（float16 + 压缩） ===
np.savez_compressed(INDEX_PATH, embeddings=embeddings, paths=json.dumps(paths))
print(f"✅ Saved small index: {INDEX_PATH} ({len(paths)} images, dim={D}, dtype={embeddings.dtype})")
