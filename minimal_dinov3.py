# Imports
import random
import torch
import numpy as np
from PIL import Image
import os, csv
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path
import matplotlib.pyplot as plt

# from torchmetrics.retrieval import RetrievalMAP
from utils.metrics import eval_func, euclidean_distance

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set to preferred GPU
MODEL_ID = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
IMAGES_ROOT = "petface/dog"
GALLERY_CSV = "splits_petface_test_10k/test_gallery.csv"
QUERY_CSV = "splits_petface_test_10k/test_query.csv"
BATCH_SIZE = 32
TOP_K = 10

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load model + processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()


# Load image paths and labels from CSV
def load_split(csv_path):
    paths, labels = [], []
    root = Path(IMAGES_ROOT)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel, pid = row.get("img_rel_path"), row.get("pid")
            if rel is None or pid is None:
                continue
            paths.append(str(root / rel))
            labels.append(int(pid))
    return paths, labels


gallery_paths, gallery_labels = load_split(GALLERY_CSV)
query_paths, query_labels = load_split(QUERY_CSV)
print(f"Loaded {len(gallery_paths)} gallery and {len(query_paths)} query images.")


# Feature (embedding) extraction using DINO
@torch.no_grad()
def extract_features(img_paths):
    feats, keep_idx = [], []
    for i in tqdm(range(0, len(img_paths), BATCH_SIZE)):
        batch = img_paths[i : i + BATCH_SIZE]
        images, idxs = [], []
        for j, p in enumerate(batch):
            try:
                images.append(Image.open(p).convert("RGB"))
                idxs.append(i + j)
            except Exception as e:
                print(f"Failed to open {p}: {e}")
        if not images:
            continue
        x = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        out = model(x)
        feat = out.last_hidden_state[:, 0, :]  # DINOv3 [CLS] token
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feats.append(feat.cpu().numpy())
        keep_idx.extend(idxs)
    if not feats:
        return np.zeros((0, 0), np.float32), []
    return np.vstack(feats), keep_idx


q_feats, q_idx = extract_features(query_paths)
g_feats, g_idx = extract_features(gallery_paths)
query_labels = [query_labels[i] for i in q_idx]
query_paths = [query_paths[i] for i in q_idx]
gallery_labels = [gallery_labels[i] for i in g_idx]
gallery_paths = [gallery_paths[i] for i in g_idx]

# Cosine similarity and ranking
pred_topk = []
for q, qlab, qpath in zip(q_feats, query_labels, query_paths):
    sims = np.dot(g_feats, q)
    for gi, gpath in enumerate(gallery_paths):
        if gpath == qpath:
            sims[gi] = -np.inf
    idx = np.argsort(sims)[-TOP_K:][::-1]
    pred_topk.append([gallery_labels[i] for i in idx])

# Top1 / Top5 / Top10 accuracy
top1 = np.mean([qlab == p[0] for qlab, p in zip(query_labels, pred_topk)])
top5 = np.mean([qlab in p[:5] for qlab, p in zip(query_labels, pred_topk)])
top10 = np.mean([qlab in p[:10] for qlab, p in zip(query_labels, pred_topk)])
print(f"Top-1: {top1:.4f}  Top-5: {top5:.4f}  Top-10: {top10:.4f}")

""" # mAP using torchmetrics
all_preds = []
all_targets = []
all_indexes = []
for q_idx, (q, qlab, qpath) in enumerate(zip(q_feats, query_labels, query_paths)):
    sims = np.dot(g_feats, q)
    for gi, gpath in enumerate(gallery_paths):
        if gpath == qpath:
            sims[gi] = -np.inf
    for gi, g_lab in enumerate(gallery_labels):
        all_preds.append(sims[gi])
        all_targets.append(int(qlab == g_lab))
        all_indexes.append(q_idx)

all_preds = torch.tensor(all_preds, dtype=torch.float32)
all_targets = torch.tensor(all_targets, dtype=torch.int)
all_indexes = torch.tensor(all_indexes, dtype=torch.long)

metric = RetrievalMAP()
mAP = metric(all_preds, all_targets, indexes=all_indexes)
print(f"mAP: {mAP.item():.4f}") """

# mAP using utils/metrics.py
qf = torch.tensor(q_feats, dtype=torch.float32)
gf = torch.tensor(g_feats, dtype=torch.float32)
q_pids = np.array(query_labels)
g_pids = np.array(gallery_labels)
q_camids = np.zeros_like(q_pids)
g_camids = np.zeros_like(g_pids)
cmc, mAP = eval_func(
    distmat=euclidean_distance(qf, gf),
    q_pids=q_pids,
    g_pids=g_pids,
    q_camids=q_camids,
    g_camids=g_camids,
    max_rank=TOP_K,
)
print(f"mAP: {mAP:.4f}")

""" # CMC Curve
cmc = np.zeros(TOP_K)
valid_q_count = 0
for qlab, preds in zip(query_labels, pred_topk):
    if qlab == -1:
        continue
    valid_q_count += 1
    found = False
    for r in range(TOP_K):
        if qlab in preds[: r + 1]:
            cmc[r:] += 1
            found = True
            break
if valid_q_count > 0:
    cmc = cmc / valid_q_count
plt.figure()
plt.plot(range(1, TOP_K + 1), cmc, marker="o")
plt.xlabel("Rank-k")
plt.ylabel("CMC")
plt.title("CMC Curve (up to Rank-10)")
plt.grid()
plt.savefig("minimal_dinov3_cmc_curve.png") """

# CMC curve from eval_func
plt.figure()
plt.plot(range(1, TOP_K + 1), cmc[:TOP_K], marker="o")
plt.xlabel("Rank-k")
plt.ylabel("CMC")
plt.title("CMC Curve (up to Rank-10)")
plt.grid()
plt.savefig("minimal_dinov3_cmc_curve.png")
