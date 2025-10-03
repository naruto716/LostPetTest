# train_petface_landmark.py
import os, time, argparse, importlib.util
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# 你的工程内模块
from datasets.petface_with_kpts import PetFaceWithKpts
from processor.landmark_align import simple_align

# ===== Config loader =====
def _load_py_cfg(cfg_path: str):
    spec = importlib.util.spec_from_file_location("user_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "CFG"):
        return dict(mod.CFG)
    raise ValueError(f"{cfg_path}  CFG")

def _merge(a: dict, b: dict):
    out = dict(a)
    out.update({k: v for k, v in b.items() if v is not None})
    return out

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="path to a python cfg file that defines CFG")
    # 可选覆盖项
    p.add_argument("--img-root", type=str)
    p.add_argument("--train-csv", type=str)
    p.add_argument("--val-query", type=str)
    p.add_argument("--val-gallery", type=str)
    p.add_argument("--test-query", type=str)
    p.add_argument("--test-gallery", type=str)
    p.add_argument("--landmarks-json", type=str)
    p.add_argument("--out", type=str)
    return p.parse_args()

def load_cfg_from_any():
    args = _parse_args()
    base = _load_py_cfg(args.config)
    override = dict(
        IMG_ROOT=args.img_root,
        TRAIN_CSV=args.train_csv,
        VAL_QUERY_CSV=args.val_query,
        VAL_GALLERY_CSV=args.val_gallery,
        TEST_QUERY_CSV=args.test_query,
        TEST_GALLERY_CSV=args.test_gallery,
        LANDMARKS_JSON=args.landmarks_json,
    )
    if args.out:
        override["OUT_DIR"] = args.out
        override["OUTPUT"] = args.out

    cfg = _merge(base, override)

    # 合法化与默认值
    out_dir = cfg.get("OUT_DIR") or cfg.get("OUTPUT") or os.path.join("outputs", "petface_landmark")
    os.makedirs(out_dir, exist_ok=True)
    cfg["OUT_DIR"] = out_dir
    cfg["OUTPUT"] = out_dir

    # 默认值兜底：防止 KeyError
    cfg.setdefault("BACKBONE", "dinov3s")
    cfg.setdefault("LR", 3.5e-4)
    cfg.setdefault("EPOCHS", 10)
    cfg.setdefault("BATCH_SIZE", 64)
    cfg.setdefault("IMG_SIZE", 256)  # 也可以给 (256,256)

    # 必需键校验
    required = ["IMG_ROOT", "TRAIN_CSV", "IMG_SIZE", "BATCH_SIZE"]
    for k in required:
        if k not in cfg or cfg[k] in (None, ""):
            raise KeyError(f"配置缺少必需键: {k}")

    return cfg

# ===== Transform：对齐后转 tensor =====
class AlignThenToTensor:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.to_tensor = transforms.ToTensor()
    def __call__(self, sample_img_pil, landmarks=None):
        aligned = simple_align(sample_img_pil, landmarks, out_size=self.size)
        return self.to_tensor(aligned)

# ===== DataLoader =====
def _norm_size(img_size):
    """允许 img_size 是 int 或 (h,w)"""
    if isinstance(img_size, int):
        return (img_size, img_size)
    if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        return (int(img_size[0]), int(img_size[1]))
    raise ValueError(f"IMG_SIZE 不合法: {img_size}")

def make_loader(img_root, split_csv, landmarks_json, img_size, batch_size, shuffle):
    size = _norm_size(img_size)
    ds = PetFaceWithKpts(
        img_root=img_root,
        split_csv=split_csv,
        landmarks_json=landmarks_json,
        transform=None  # 在 collate 里做对齐+tensor
    )
    def collate(batch):
        xs, ys = [], []
        toT = AlignThenToTensor(size)
        for b in batch:
            x = toT(b["image"], b.get("landmarks"))
            xs.append(x)
            ys.append(int(b["id"]))  # id 为 "000007" 也能转成 int
        return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate, pin_memory=True)

# ===== Model =====
def build_model(num_classes, backbone_name="dinov3s"):
    from model.backbones import build_backbone
    backbone, feat_dim = build_backbone(backbone_name)
    head = nn.Linear(feat_dim, num_classes)
    return nn.Sequential(backbone, nn.Flatten(1), head)

def count_classes(csv_path):
    import csv
    ids = set()
    with open(csv_path) as f:
        r = csv.reader(f)
        header = next(r, None)
        # 兼容两种：有表头('img,id') 或无表头
        def parse_id(val):
            try:
                return int(val)
            except:
                return None
        if header and header[0] == "img":
            for row in r:
                if len(row) >= 2:
                    x = parse_id(row[1])
                    if x is not None:
                        ids.add(x)
        else:
            # 第一行不是表头，先处理它
            if header and len(header) >= 2:
                x = parse_id(header[1])
                if x is not None:
                    ids.add(x)
            for row in r:
                if len(row) >= 2:
                    x = parse_id(row[1])
                    if x is not None:
                        ids.add(x)
    return max(len(ids), 1)

def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / max(total_n, 1)

@torch.no_grad()
def eval_top1(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total > 0 else 0.0


def main():
    cfg = load_cfg_from_any()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== 打印基础信息 ====
    print("="*60)
    print("🎯 Starting Landmark Training")
    print(f"📁 Output: {cfg['OUT_DIR']}")
    print(f"🧠 Backbone: {cfg['BACKBONE']}")
    print(f"🔧 Backbone Frozen:", bool(cfg.get("FREEZE_BACKBONE", False)))
    print("="*60)

    # ==== Data ====
    train_loader = make_loader(
        cfg["IMG_ROOT"], cfg["TRAIN_CSV"], cfg.get("LANDMARKS_JSON"),
        cfg["IMG_SIZE"], cfg["BATCH_SIZE"], shuffle=True
    )
    # 简单 sanity：用 val_query 当分类验证集（后续可换成真正 ReID 评测）
    val_loader = make_loader(
        cfg["IMG_ROOT"], cfg.get("VAL_QUERY_CSV", cfg["TRAIN_CSV"]),
        cfg.get("LANDMARKS_JSON"), cfg["IMG_SIZE"], cfg["BATCH_SIZE"], shuffle=False
    )

    # ==== Model ====
    num_classes = count_classes(cfg["TRAIN_CSV"])
    model = build_model(num_classes, cfg["BACKBONE"]).to(device)

    # 可选：冻结骨干
    if cfg.get("FREEZE_BACKBONE", False):
        # 假设 model = nn.Sequential(backbone, Flatten, head)
        for p in model[0].parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["LR"])
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    # 记录一次 cfg 以便复现
    with open(os.path.join(cfg["OUT_DIR"], "run_cfg.txt"), "w") as f:
        for k, v in sorted(cfg.items()):
            f.write(f"{k}={v}\n")

    # ==== Train ====
    best = 0.0
    epochs = int(cfg["EPOCHS"])
    for ep in range(epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, device)
        top1 = eval_top1(model, val_loader, device)
        dt = time.time() - t0
        print(f"[{ep+1}/{epochs}] loss={tr_loss:.4f} top1={top1:.3f} time={dt:.1f}s")

        # 保存 last
        torch.save(model.state_dict(), os.path.join(cfg["OUT_DIR"], "last.pth"))
        # 更新 best
        if top1 > best:
            best = top1
            torch.save(model.state_dict(), os.path.join(cfg["OUT_DIR"], "best.pth"))

    print("="*60)
    print("✅ Finished. Best top1:", best)




if __name__ == "__main__":
    main()
