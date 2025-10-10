"""
Dog Re-Identification with DINO backbone using ArcFace loss (classification batch training).
- ArcFace implementation is fixed, but low performance, needs further tuning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import pandas as pd
import re
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from pytorch_metric_learning import losses
import random

CROP_FEATURES = True
SEED = 42
ARCFACE = True
BATCH_SIZE = 32
NUM_EPOCHS = 10

class FusionNet(nn.Module):
    def __init__(self, crop_dim, dino_dim, out_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(crop_dim + dino_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )
    def forward(self, crop_feat, dino_feat):
        x = torch.cat([crop_feat, dino_feat], dim=-1)
        return self.fc(x)

class CropEmbeddingNet(nn.Module):
    def __init__(self, input_dim=3 * 224 * 224, embed_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )
    def forward(self, x):
        return self.fc(x)

def get_full_image_map(split_csv):
    df = pd.read_csv(split_csv)
    img_map = {}
    for _, row in df.iterrows():
        pid = int(row["pid"])
        m = re.match(r"(\d+)/(\d+)\.png", row["img_rel_path"])
        if m:
            pid_str, imgidx = m.groups()
            img_map[(pid, imgidx)] = row["img_rel_path"]
    return img_map

def group_crops_by_pid_imgidx(crop_csv):
    df = pd.read_csv(crop_csv)
    def parse_imgidx(path):
        m = re.match(r"(\d+)_(\d+)_([a-z_]+)\.png", path)
        if m:
            pid, imgidx, region = m.groups()
            return pid, imgidx, region
        return None, None, None
    groups = {}
    for _, row in df.iterrows():
        pid, imgidx, region = parse_imgidx(row["path"])
        if pid is None:
            continue
        key = (int(pid), imgidx)
        if key not in groups:
            groups[key] = []
        groups[key].append({"path": row["path"], "region": region})
    return groups

class ReIDClassificationDataset(Dataset):
    def __init__(self, crop_groups, img_map, crop_transform, num_crops, crop_embedder, fusion_net, device, pid_to_label):
        self.keys = list(crop_groups.keys())
        self.crop_groups = crop_groups
        self.img_map = img_map
        self.crop_transform = crop_transform
        self.num_crops = num_crops
        self.crop_embedder = crop_embedder
        self.fusion_net = fusion_net
        self.device = device
        self.pid_to_label = pid_to_label
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        key = self.keys[idx]
        crops = self.crop_groups[key]
        crop_tensors = []
        for crop_info in crops:
            crop_img = Image.open(f"output/train_crops/{crop_info['path']}").convert("RGB")
            crop_tensor = self.crop_transform(crop_img).unsqueeze(0)
            crop_tensors.append(crop_tensor)
        if not crop_tensors:
            crop_embeds = torch.zeros(self.num_crops, 128, device=self.device)
        else:
            crop_tensors = torch.cat(crop_tensors, dim=0)
            crop_embeds = self.crop_embedder(crop_tensors.to(self.device))
            n = crop_embeds.shape[0]
            if n < self.num_crops:
                pad = torch.zeros(self.num_crops - n, 128, device=self.device)
                crop_embeds = torch.cat([crop_embeds, pad], dim=0)
            elif n > self.num_crops:
                crop_embeds = crop_embeds[:self.num_crops]
        if CROP_FEATURES:
            concat_embed = crop_embeds.flatten()
        else:
            concat_embed = torch.zeros_like(crop_embeds.flatten(), device=self.device)
        img_rel_path = self.img_map[key]
        m = re.match(r"(\d+)/(\d+)\.png", img_rel_path)
        if m:
            pid_str, imgidx = m.groups()
            flat_img_name = f"{pid_str}_{imgidx}.png"
            full_img_path = f"output/train/{flat_img_name}"
        else:
            raise FileNotFoundError(f"Could not parse image path: {img_rel_path}")
        full_img = Image.open(full_img_path).convert("RGB")
        full_img_tensor = self.crop_transform(full_img).unsqueeze(0)
        with torch.no_grad():
            outputs = dino_model(full_img_tensor.to(self.device))
            dino_feat = outputs.last_hidden_state[:, 0].flatten()
        if concat_embed.dim() == 1:
            concat_embed = concat_embed.unsqueeze(0)
        if dino_feat.dim() == 1:
            dino_feat = dino_feat.unsqueeze(0)
        fused = self.fusion_net(concat_embed, dino_feat)
        label = self.pid_to_label[key[0]]
        return fused.squeeze(0), label  # Do NOT move to device here

def evaluate_cmc_map(query_features, query_pids, gallery_features, gallery_pids, topk=(1, 5, 10)):
    distmat = torch.cdist(query_features, gallery_features, p=2).cpu().numpy()
    query_pids = np.array(query_pids)
    gallery_pids = np.array(gallery_pids)
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, None])
    cmc = np.zeros(num_g)
    all_AP = []
    for i in range(num_q):
        m = matches[i]
        if not np.any(m):
            continue
        rank_idx = np.where(m)[0][0]
        cmc[rank_idx:] += 1
        y_true = m.astype(np.int32)
        y_score = -distmat[i][indices[i]]
        num_rel = y_true.sum()
        tmp_cmc = y_true.cumsum()
        precision = tmp_cmc / (np.arange(len(y_true)) + 1)
        AP = (precision * y_true).sum() / num_rel if num_rel > 0 else 0
        all_AP.append(AP)
    cmc = cmc / num_q
    mAP = np.mean(all_AP) if all_AP else 0.0
    print("[EVAL] CMC:")
    for k in topk:
        print(f"  Rank-{k}: {cmc[k-1]*100:.2f}%")
    print(f"[EVAL] mAP: {mAP*100:.2f}%")

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    crop_groups = group_crops_by_pid_imgidx("output/train_crops.csv")
    img_map = get_full_image_map("subset_splits_petface/train_subset.csv")
    crop_embedder = CropEmbeddingNet().to(device)
    crop_embedder.eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    dino_model.eval()
    num_crops = 6
    crop_dim = 128 * num_crops
    dino_dim = 768
    fusion_net = FusionNet(crop_dim, dino_dim).to(device)
    fusion_net.eval()
    unique_pids = sorted(set(pid for pid, imgidx in img_map.keys()))
    pid_to_label = {pid: idx for idx, pid in enumerate(unique_pids)}
    num_classes = len(unique_pids)
    loss_func = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=256)
    optimizer = torch.optim.Adam(list(crop_embedder.parameters()) + list(fusion_net.parameters()), lr=1e-3)
    classification_dataset = ReIDClassificationDataset(
        crop_groups, img_map, crop_transform, num_crops, crop_embedder, fusion_net, device, pid_to_label
    )
    classification_loader = DataLoader(classification_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(NUM_EPOCHS):
        crop_embedder.train()
        fusion_net.train()
        for feats, labels in classification_loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = loss_func(feats, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[TRAIN][Epoch {epoch+1}] ArcFace Loss: {loss.item():.4f}")

    # Evaluation phase
    crop_embedder.eval()
    fusion_net.eval()
    dino_model.eval()
    def extract_features(csv_path, crop_dir, img_map_csv, img_dir):
        crop_groups = group_crops_by_pid_imgidx(csv_path)
        img_map = get_full_image_map(img_map_csv)
        features = []
        pids = []
        img_indices = []
        with torch.no_grad():
            for key, crops in crop_groups.items():
                if key not in img_map:
                    continue
                crop_tensors = []
                for crop_info in crops:
                    crop_img = Image.open(f"{crop_dir}/{crop_info['path']}").convert("RGB")
                    crop_tensor = crop_transform(crop_img).unsqueeze(0)
                    crop_tensors.append(crop_tensor)
                if not crop_tensors:
                    crop_embeds = torch.zeros(num_crops, 128)
                else:
                    crop_tensors = torch.cat(crop_tensors, dim=0)
                    crop_embeds = crop_embedder(crop_tensors.to(device))
                    n = crop_embeds.shape[0]
                    if n < num_crops:
                        pad = torch.zeros(num_crops - n, 128, device=device)
                        crop_embeds = torch.cat([crop_embeds, pad], dim=0)
                    elif n > num_crops:
                        crop_embeds = crop_embeds[:num_crops]
                if CROP_FEATURES:
                    concat_embed = crop_embeds.flatten()
                else:
                    concat_embed = torch.zeros_like(crop_embeds.flatten(), device=device)
                img_rel_path = img_map[key]
                m = re.match(r"(\d+)/(\d+)\.png", img_rel_path)
                if m:
                    pid_str, imgidx = m.groups()
                    flat_img_name = f"{pid_str}_{imgidx}.png"
                    full_img_path = f"{img_dir}/{flat_img_name}"
                else:
                    raise FileNotFoundError(f"Could not parse image path: {img_rel_path}")
                full_img = Image.open(full_img_path).convert("RGB")
                full_img_tensor = crop_transform(full_img).unsqueeze(0)
                with torch.no_grad():
                    outputs = dino_model(full_img_tensor.to(device))
                    dino_feat = outputs.last_hidden_state[:, 0].flatten()
                if concat_embed.dim() == 1:
                    concat_embed = concat_embed.unsqueeze(0)
                if dino_feat.dim() == 1:
                    dino_feat = dino_feat.unsqueeze(0)
                fused = fusion_net(concat_embed.to(device), dino_feat)
                features.append(fused.cpu())
                pids.append(key[0])
                img_indices.append(key[1])
        features = torch.cat(features, dim=0)
        return features, pids, img_indices

    # Extract features for query and gallery
    query_features, query_pids, query_imgidx = extract_features(
        "output/query_crops.csv", "output/query_crops", "subset_splits_petface/test_query_subset.csv", "output/query"
    )
    gallery_features, gallery_pids, gallery_imgidx = extract_features(
        "output/gallery_crops.csv", "output/gallery_crops", "subset_splits_petface/test_gallery_subset.csv", "output/gallery"
    )
    print(f"[EVAL] Query features: {query_features.shape}, Gallery features: {gallery_features.shape}")

    # Metrics computation
    num_individuals = len(set(pid for pid, imgidx in img_map.keys()))
    print("Model: DINOv2, Number of unique individuals:", num_individuals)
    print("Crop features enabled:", CROP_FEATURES, ",Epochs:", NUM_EPOCHS, ", ArcFace Loss")
    evaluate_cmc_map(query_features, query_pids, gallery_features, gallery_pids)
