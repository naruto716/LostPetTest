#!/usr/bin/env python
"""
Launcher that composes configuration overrides and calls the actual training script.

Features
--------
- Accepts a small set of "runner" arguments (output_dir, epochs, eval_only, resume, eval_split).
- Accepts common config overrides (e.g., base_lr, triplet_margin, bn_neck, etc.).
- Passes through any unknown flags directly to `RegionalConfig.from_cli`, so every
  hyperparameter can be overridden without changing this file.
- Writes the constructed config back to `config_regional.cfg`, ensuring a single
  source of truth for `train_regional`.
"""

import argparse
import sys


def str2bool(x):
    s = str(x).lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {x}")


def main():
    # ---------------- Runner arguments (handled by the launcher) ----------------
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--output_dir", type=str, default=None, help="Training output directory.")
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    p.add_argument("--eval_only", action="store_true", help="Run evaluation only.")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    p.add_argument("--eval_split", choices=["val", "test"], default="val", help="Evaluation split.")

    # ---------------- Common config overrides (backward compatible) -------------
    p.add_argument("--base_lr", type=float, default=None, help="Base learning rate.")
    p.add_argument("--triplet_margin", type=float, default=None, help="Triplet loss margin.")
    p.add_argument("--triplet_w", type=float, default=None, help="Triplet loss weight.")
    p.add_argument("--id_w", type=float, default=None, help="ID (classification) loss weight.")
    p.add_argument("--bn_neck", type=str, default=None, help="Enable BN neck (true/false).")
    p.add_argument("--label_smoothing", type=float, default=None, help="Label smoothing factor.")
    p.add_argument("--use_center_loss", type=str, default=None, help="Enable center loss (true/false).")
    p.add_argument("--center_w", type=float, default=None, help="Center loss weight.")
    p.add_argument("--freeze_backbone", type=str, default=None, help="Freeze backbone (true/false).")
    p.add_argument("--ims_per_batch", type=int, default=None, help="Images per batch.")
    p.add_argument("--num_instance", type=int, default=None, help="Images per identity (K).")
    p.add_argument("--test_query_split", type=str, default=None, help="Test query CSV path.")
    p.add_argument("--test_gallery_split", type=str, default=None, help="Test gallery CSV path.")

    # Any unknown arguments are passed to `RegionalConfig.from_cli`.
    args, rest = p.parse_known_args()

    # ---------------- Build list for RegionalConfig.from_cli --------------------
    cfg_args = list(rest)  # pass-through for any extra/unknown flags

    def add(k, v):
        """Append a flag to cfg_args if a value was provided."""
        if v is None:
            return
        if isinstance(v, bool):
            cfg_args.extend([f"--{k}", "true" if v else "false"])
        elif isinstance(v, (list, tuple)):
            cfg_args.append(f"--{k}")
            cfg_args.extend([str(x) for x in v])
        else:
            cfg_args.extend([f"--{k}", str(v)])

    # Forward common overrides to the config layer
    add("base_lr", args.base_lr)
    add("triplet_margin", args.triplet_margin)
    add("label_smoothing", args.label_smoothing)
    add("bn_neck", args.bn_neck)                # string "true"/"false" -> parsed in config
    add("freeze_backbone", args.freeze_backbone)
    add("ims_per_batch", args.ims_per_batch)
    add("num_instance", args.num_instance)
    add("test_query_split", args.test_query_split)
    add("test_gallery_split", args.test_gallery_split)

    # ---------------- Create cfg via RegionalConfig.from_cli --------------------
    import config_regional as C
    try:
        RegionalConfig = getattr(C, "RegionalConfig")
        cfg = RegionalConfig.from_cli(cfg_args)
    except Exception:
        # Fallback to existing module-level cfg and apply overrides manually
        cfg = getattr(C, "cfg")

    # Manual aliases / extra weights that may not exist in the class definition
    if args.triplet_w is not None:
        setattr(cfg, "TRIPLET_LOSS_WEIGHT", args.triplet_w)
    if args.id_w is not None:
        setattr(cfg, "ID_LOSS_WEIGHT", args.id_w)
    if args.use_center_loss is not None:
        setattr(cfg, "USE_CENTER_LOSS", str2bool(args.use_center_loss))
    if args.center_w is not None:
        setattr(cfg, "CENTER_LOSS_WEIGHT", args.center_w)
        setattr(cfg, "CENTER_W", args.center_w)  # support either name

    # Output dir / epochs override (common to runner and config)
    if args.output_dir is not None:
        setattr(cfg, "OUTPUT_DIR", args.output_dir)
    if args.epochs is not None:
        setattr(cfg, "MAX_EPOCHS", args.epochs)

    # Expose the constructed config back to the module for training imports
    C.cfg = cfg

    # ---------------- Hand off to the training script ---------------------------
    # Only pass runner arguments that the training script expects.
    sys.argv = ["train_regional.py"]
    if args.output_dir:
        sys.argv += ["--output_dir", args.output_dir]
    if args.epochs:
        sys.argv += ["--epochs", str(args.epochs)]
    if args.eval_only:
        sys.argv += ["--eval_only"]
    if args.resume:
        sys.argv += ["--resume", args.resume]
    if args.eval_split:
        sys.argv += ["--eval_split", args.eval_split]

    import train_regional
    train_regional.main()


if __name__ == "__main__":
    main()
