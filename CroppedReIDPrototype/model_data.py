"""
Dog Re-Identification with DINO backbone.
Loads a trained model, extracts features, and evaluates mAP/CMC.

- Loads and preprocesses dog images, extracting both local crops (e.g., eyes, nose) and full images.
- Uses a simple neural network (CropEmbeddingNet) to embed each crop into a feature vector.
- Uses a pretrained DINOv2 vision transformer to extract a global feature from the full image ([CLS] token).
- Concatenates all crop features and the DINOv2 feature, then fuses them using a FusionNet to produce a single embedding per image.
- Trains the model using triplet loss, encouraging embeddings of the same individual to be closer than those of different individuals.
- After training, extracts fused features for all query and gallery images.
- Computes CMC and mAP metrics to evaluate how well the model retrieves correct matches in a re-identification scenario.
"""

import torch
from dataset_loader import PreCroppedMultiPartReIDDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import pandas as pd
import re
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import random
from pytorch_metric_learning import losses

CROP_FEATURES = True  # Set to False to disable crop features
SEED = 42
ARCFACE = False  # DO NOT SET TO TRUE | Arcface implementation not working properly

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
    """
    Returns a dict mapping (pid, imgidx) -> img_rel_path for full images.
    """
    df = pd.read_csv(split_csv)
    img_map = {}
    for _, row in df.iterrows():
        pid = int(row["pid"])
        # img_rel_path is like 050780/00.png
        m = re.match(r"(\d+)/(\d+)\.png", row["img_rel_path"])
        if m:
            pid_str, imgidx = m.groups()
            img_map[(pid, imgidx)] = row["img_rel_path"]
    return img_map


def run_dino_on_crops(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_features, all_pids, all_regions = [], [], []
    with torch.no_grad():
        for crops, pids, regions in loader:
            crops = crops.to(device)
            feats = model(crops, return_mode="features")
            all_features.append(feats.cpu())
            all_pids.extend(pids)
            all_regions.extend(regions)
    all_features = torch.cat(all_features, dim=0)
    return all_features, all_pids, all_regions


def group_crops_by_pid_imgidx(crop_csv):
    df = pd.read_csv(crop_csv)

    # Extract image index from filename (e.g., 050780_00_left_eye.png)
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

def evaluate_cmc_map(query_features, query_pids, gallery_features, gallery_pids, topk=(1, 5, 10)):
        # Compute distance matrix (Euclidean)
        distmat = torch.cdist(query_features, gallery_features, p=2).cpu().numpy()
        query_pids = np.array(query_pids)
        gallery_pids = np.array(gallery_pids)
        num_q, num_g = distmat.shape
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_pids[indices] == query_pids[:, None])

        # CMC
        cmc = np.zeros(num_g)
        all_AP = []
        for i in range(num_q):
            m = matches[i]
            if not np.any(m):
                continue
            rank_idx = np.where(m)[0][0]
            cmc[rank_idx:] += 1
            # mAP
            y_true = m.astype(np.int32)
            y_score = -distmat[i][indices[i]]
            # AP calculation
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

# -----
# MAIN
# -----
if __name__ == "__main__":

    # Set seeds
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup model (replace with your actual model loading code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # from model import make_model
    # model = make_model(...)
    # model.to(device)

    # Prepare dataset and dataloader
    crop_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    model_data = PreCroppedMultiPartReIDDataset(
        split_csv="output/train_crops.csv",
        crop_dir="output/train_crops",
        transform=crop_transform,
    )

    # Group crops by (pid, image index)
    crop_groups = group_crops_by_pid_imgidx("output/train_crops.csv")
    print(f"Number of (pid, imgidx) groups: {len(crop_groups)}")

    # Map (pid, imgidx) to full image path
    img_map = get_full_image_map("subset_splits_petface/train_subset.csv")
    print(f"Number of full image entries: {len(img_map)}")

    # Crop embedder
    crop_embedder = CropEmbeddingNet().to(device)
    crop_embedder.eval()

    # DINOv2 model from Hugging Face
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    dino_model.eval()

    def get_dinov2_features(pil_img):
        """
        pil_img: PIL.Image (RGB)
        Returns: torch.Tensor of shape (1, hidden_dim) for [CLS] token
        """
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            features = outputs.last_hidden_state[:, 0]  # [CLS] token
        return features

    # Fusion net (output 256-dim vector)
    # For demonstration, assume crop_embedder outputs 128-dim per crop, 6 crops per group, DINO 768-dim
    num_crops = 6
    crop_dim = 128 * num_crops
    dino_dim = 768  # DINOv2 ViT-Base [CLS] token is 768-dim
    fusion_net = FusionNet(crop_dim, dino_dim).to(device)
    fusion_net.eval()

    # ArcFace loss (optional)
    num_classes = len(set(pid for pid, imgidx in img_map.keys()))
    if ARCFACE:
        # Build pid-to-label mapping for zero-based contiguous labels
        unique_pids = sorted(set(pid for pid, imgidx in img_map.keys()))
        pid_to_label = {pid: idx for idx, pid in enumerate(unique_pids)}
        loss_func = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=256)
    else:
        criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # Example: Training loop
    optimizer = torch.optim.Adam(list(crop_embedder.parameters()) + list(fusion_net.parameters()), lr=1e-3)
    triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # Build a list of (pid, imgidx) keys for sampling
    keys_by_pid = {}
    for key in crop_groups:
        pid = key[0]
        if pid not in keys_by_pid:
            keys_by_pid[pid] = []
        keys_by_pid[pid].append(key)

    # EPOCH-BASED TRAINING PHASE (Triplet Loss)
    num_epochs = 10  # Set your desired number of epochs
    num_triplets_per_epoch = 100  # Set triplets per epoch
    pids = list(keys_by_pid.keys())

    for epoch in range(num_epochs):
        crop_embedder.train()
        fusion_net.train()
        triplet_count = 0
        while triplet_count < num_triplets_per_epoch:
            anchor_pid = random.choice(pids)
            if len(keys_by_pid[anchor_pid]) < 2:
                continue  # Need at least 2 images for anchor/positive
            anchor_key, positive_key = random.sample(keys_by_pid[anchor_pid], 2)
            negative_pid = random.choice([pid for pid in pids if pid != anchor_pid])
            negative_key = random.choice(keys_by_pid[negative_pid])

            def get_fused(key):
                crops = crop_groups[key]
                crop_tensors = []
                for crop_info in crops:
                    crop_img = Image.open(f"output/train_crops/{crop_info['path']}").convert("RGB")
                    crop_tensor = crop_transform(crop_img).unsqueeze(0).to(device)
                    crop_tensors.append(crop_tensor)
                if not crop_tensors:
                    crop_embeds = torch.zeros(num_crops, 128, device=device)
                else:
                    crop_tensors = torch.cat(crop_tensors, dim=0)
                    crop_embeds = crop_embedder(crop_tensors)
                    n = crop_embeds.shape[0]
                    if n < num_crops:
                        pad = torch.zeros(num_crops - n, 128, device=device)
                        crop_embeds = torch.cat([crop_embeds, pad], dim=0)
                    elif n > num_crops:
                        crop_embeds = crop_embeds[:num_crops]
                # CHANGE HERE FOR TESTING CROPS VS NO CROPS
                if CROP_FEATURES:
                    concat_embed = crop_embeds.flatten()
                else:
                    concat_embed = torch.zeros_like(crop_embeds.flatten())
                img_rel_path = img_map[key]
                m = re.match(r"(\d+)/(\d+)\.png", img_rel_path)
                if m:
                    pid_str, imgidx = m.groups()
                    flat_img_name = f"{pid_str}_{imgidx}.png"
                    full_img_path = f"output/train/{flat_img_name}"
                else:
                    raise FileNotFoundError(f"Could not parse image path: {img_rel_path}")
                full_img = Image.open(full_img_path).convert("RGB")
                full_img_tensor = crop_transform(full_img).unsqueeze(0).to(device)
                # Use the [CLS] token from last_hidden_state as DINO feature
                with torch.no_grad():
                    outputs = dino_model(full_img_tensor)
                    dino_feat = outputs.last_hidden_state[:, 0].flatten()
                # Ensure both concat_embed and dino_feat are 2D for FusionNet
                if concat_embed.dim() == 1:
                    concat_embed = concat_embed.unsqueeze(0)
                if dino_feat.dim() == 1:
                    dino_feat = dino_feat.unsqueeze(0)
                fused = fusion_net(concat_embed, dino_feat)
                return fused.squeeze(0)

            anchor_feat = get_fused(anchor_key)
            positive_feat = get_fused(positive_key)
            negative_feat = get_fused(negative_key)

            if ARCFACE:
                # Map pids to zero-based labels for ArcFace
                labels = torch.tensor([
                    pid_to_label[anchor_pid],
                    pid_to_label[anchor_pid],
                    pid_to_label[negative_pid]
                ], device=device)
                feats = torch.stack([anchor_feat, positive_feat, negative_feat], dim=0)
                loss = loss_func(feats, labels)
            else:
                loss = criterion(anchor_feat, positive_feat, negative_feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (triplet_count+1) % 10 == 0 or triplet_count == 0:
                print(f"[TRAIN][Epoch {epoch+1}] Triplet {triplet_count+1}/{num_triplets_per_epoch} | Loss: {loss.item():.4f}")
            triplet_count += 1

    # EVALUATION PHASE (Query & Gallery)
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
                    crop_tensor = crop_transform(crop_img).unsqueeze(0).to(device)
                    crop_tensors.append(crop_tensor)
                if not crop_tensors:
                    crop_embeds = torch.zeros(num_crops, 128, device=device)
                else:
                    crop_tensors = torch.cat(crop_tensors, dim=0)
                    crop_embeds = crop_embedder(crop_tensors)
                    n = crop_embeds.shape[0]
                    if n < num_crops:
                        pad = torch.zeros(num_crops - n, 128, device=device)
                        crop_embeds = torch.cat([crop_embeds, pad], dim=0)
                    elif n > num_crops:
                        crop_embeds = crop_embeds[:num_crops]
                concat_embed = crop_embeds.flatten()
                img_rel_path = img_map[key]
                m = re.match(r"(\d+)/(\d+)\.png", img_rel_path)
                if m:
                    pid_str, imgidx = m.groups()
                    flat_img_name = f"{pid_str}_{imgidx}.png"
                    full_img_path = f"{img_dir}/{flat_img_name}"
                else:
                    raise FileNotFoundError(f"Could not parse image path: {img_rel_path}")
                full_img = Image.open(full_img_path).convert("RGB")
                full_img_tensor = crop_transform(full_img).unsqueeze(0).to(device)
                dino_feat = dino_model(full_img_tensor).last_hidden_state[:, 0]

                # Ensure dino_feat is 2D for concatenation
                if dino_feat.dim() == 1:
                    dino_feat = dino_feat.unsqueeze(0)
                fused = fusion_net(concat_embed.unsqueeze(0), dino_feat)
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
    print("Crop features enabled:", CROP_FEATURES, ",Epochs:", num_epochs, ",Triplets per epoch:", num_triplets_per_epoch)
    evaluate_cmc_map(query_features, query_pids, gallery_features, gallery_pids)