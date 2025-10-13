# launch.py
import argparse, sys
from pathlib import Path

def str2bool(x):
    return str(x).lower() in {"1","true","t","yes","y"}

def main():
    p = argparse.ArgumentParser()
    # 训练脚本已有的参数（透传）
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--resume", type=str, default=None)

    # 我们新增的可覆盖项（直接改 cfg）
    p.add_argument("--base_lr", type=float, default=None)
    p.add_argument("--triplet_margin", type=float, default=None)
    p.add_argument("--triplet_w", type=float, default=None)
    p.add_argument("--id_w", type=float, default=None)
    p.add_argument("--bn_neck", type=str, default=None)            # true/false
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--use_center_loss", type=str, default=None)     # true/false
    p.add_argument("--center_w", type=float, default=None)
    p.add_argument("--freeze_backbone", type=str, default=None)     # true/false
    p.add_argument("--ims_per_batch", type=int, default=None)
    p.add_argument("--num_instance", type=int, default=None)
    p.add_argument("--eval_split", choices=["val","test"], default="val")
    p.add_argument("--test_query_split", type=str, default=None)
    p.add_argument("--test_gallery_split", type=str, default=None)
    args, _ = p.parse_known_args()

    # 提前改 cfg
    from config_regional import cfg
    if args.base_lr is not None: cfg.BASE_LR = args.base_lr
    if args.triplet_margin is not None: cfg.TRIPLET_MARGIN = args.triplet_margin
    if args.triplet_w is not None: cfg.TRIPLET_LOSS_WEIGHT = args.triplet_w
    if args.id_w is not None: cfg.ID_LOSS_WEIGHT = args.id_w
    if args.bn_neck is not None: cfg.BN_NECK = str2bool(args.bn_neck)
    if args.label_smoothing is not None: cfg.LABEL_SMOOTHING = args.label_smoothing
    if args.use_center_loss is not None: cfg.USE_CENTER_LOSS = str2bool(args.use_center_loss)
    if args.center_w is not None: cfg.CENTER_LOSS_WEIGHT = args.center_w
    if args.freeze_backbone is not None: cfg.FREEZE_BACKBONE = str2bool(args.freeze_backbone)
    if args.ims_per_batch is not None: cfg.IMS_PER_BATCH = args.ims_per_batch
    if args.num_instance is not None: cfg.NUM_INSTANCE = args.num_instance
    if args.output_dir is not None: cfg.OUTPUT_DIR = args.output_dir
    if args.epochs is not None: cfg.MAX_EPOCHS = args.epochs
    if args.test_query_split is not None: cfg.TEST_QUERY_SPLIT = args.test_query_split
    if args.test_gallery_split is not None: cfg.TEST_GALLERY_SPLIT = args.test_gallery_split
    # 透传给训练脚本的 argparse
    sys.argv = ["train_regional.py"]  # 训练脚本的名字（见下行 import）
    if args.output_dir: sys.argv += ["--output_dir", args.output_dir]
    if args.epochs: sys.argv += ["--epochs", str(args.epochs)]
    if args.eval_only: sys.argv += ["--eval_only"]
    if args.resume: sys.argv += ["--resume", args.resume]

    # 延迟导入并运行
    import train_regional
    train_regional.main()

if __name__ == "__main__":
    main()
