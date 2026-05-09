from __future__ import annotations

import argparse

from .paths import ensure_layout
from .yolo import DEFAULT_METHODS, DEFAULT_YOLO_VARIANT, YOLO_VARIANTS

DEFAULT_TRACKER_METHODS = [
    "yolo_balltracker",
    "yolo_sort",
    "yolo_deepsort",
    "yolo_bytetrack",
    "yolo_botsort",
    "yolo_botsort_finalized",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Balltest experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="scan dataset, split train/select/test, export datasets")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--test-ratio", type=float, default=0.15)
    prepare.add_argument("--select-ratio", type=float, default=0.15)
    prepare.add_argument("--force", action="store_true")

    train_yolo = sub.add_parser("train-yolo", help="train YOLO detector")
    train_yolo.add_argument("--variant", choices=list(YOLO_VARIANTS), default=DEFAULT_YOLO_VARIANT)
    train_yolo.add_argument("--epochs", type=int, default=160)
    train_yolo.add_argument("--batch", type=int, default=48)
    train_yolo.add_argument("--patience", type=int, default=30)
    train_yolo.add_argument("--device", default="")
    train_yolo.add_argument("--resume", action="store_true")

    cache_yolo = sub.add_parser("cache-yolo", help="cache raw YOLO detections for the test split")
    cache_yolo.add_argument("--variant", choices=list(YOLO_VARIANTS), default=DEFAULT_YOLO_VARIANT)
    cache_yolo.add_argument("--model-path", default="")
    cache_yolo.add_argument("--device", default="")
    cache_yolo.add_argument("--overwrite", action="store_true")
    cache_yolo.add_argument("--max-clips", type=int, default=0)

    train_tracknet = sub.add_parser("train-tracknet", help="train TrackNet")
    train_tracknet.add_argument("--epochs", type=int, default=150)
    train_tracknet.add_argument("--batch-size", type=int, default=2)
    train_tracknet.add_argument("--lr", type=float, default=1.0)
    train_tracknet.add_argument("--val-interval", type=int, default=1)
    train_tracknet.add_argument("--patience-rounds", type=int, default=15)
    train_tracknet.add_argument("--device", default="")
    train_tracknet.add_argument("--resume", action="store_true")

    train_tracknet_v4 = sub.add_parser("train-tracknet-v4", help="train the local TrackNetV4 implementation")
    train_tracknet_v4.add_argument("--epochs", type=int, default=150)
    train_tracknet_v4.add_argument("--batch-size", type=int, default=2)
    train_tracknet_v4.add_argument("--learning-rate", type=float, default=1.0)
    train_tracknet_v4.add_argument("--patience-rounds", type=int, default=15)
    train_tracknet_v4.add_argument("--resume", action="store_true")

    train_tracknet_v2 = sub.add_parser("train-tracknet-v2", help="train the local TrackNetV2 implementation")
    train_tracknet_v2.add_argument("--epochs", type=int, default=150)
    train_tracknet_v2.add_argument("--batch-size", type=int, default=2)
    train_tracknet_v2.add_argument("--learning-rate", type=float, default=1.0)
    train_tracknet_v2.add_argument("--patience-rounds", type=int, default=15)
    train_tracknet_v2.add_argument("--device", default="")
    train_tracknet_v2.add_argument("--resume", action="store_true")

    train_tracknet_v3 = sub.add_parser("train-tracknet-v3", help="train the local TrackNetV3 tracking module")
    train_tracknet_v3.add_argument("--epochs", type=int, default=150)
    train_tracknet_v3.add_argument("--batch-size", type=int, default=2)
    train_tracknet_v3.add_argument("--learning-rate", type=float, default=1e-3)
    train_tracknet_v3.add_argument("--mixup-prob", type=float, default=0.3)
    train_tracknet_v3.add_argument("--rectifier-epochs", type=int, default=30)
    train_tracknet_v3.add_argument("--mask-ratio", type=float, default=0.3)
    train_tracknet_v3.add_argument("--rectify-window", type=int, default=8)
    train_tracknet_v3.add_argument("--patience-rounds", type=int, default=15)
    train_tracknet_v3.add_argument("--device", default="")
    train_tracknet_v3.add_argument("--resume", action="store_true")

    train_tracknet_v5 = sub.add_parser("train-tracknet-v5", help="train the local TrackNetV5 implementation")
    train_tracknet_v5.add_argument("--epochs", type=int, default=150)
    train_tracknet_v5.add_argument("--batch-size", type=int, default=2)
    train_tracknet_v5.add_argument("--learning-rate", type=float, default=1.0)
    train_tracknet_v5.add_argument("--patience-rounds", type=int, default=15)
    train_tracknet_v5.add_argument("--device", default="")
    train_tracknet_v5.add_argument("--resume", action="store_true")

    eval_methods = sub.add_parser("eval-methods", help="compare methods on the test split")
    eval_methods.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    eval_methods.add_argument("--dist-threshold", type=float, default=15.0)
    eval_methods.add_argument("--max-clips", type=int, default=0)
    eval_methods.add_argument("--device", default="")
    eval_methods.add_argument("--yolo-variant", choices=list(YOLO_VARIANTS), default=DEFAULT_YOLO_VARIANT)
    eval_methods.add_argument("--yolo-model-path", default="")
    eval_methods.add_argument("--tracknet-public-weight", default="")

    eval_trackers = sub.add_parser("eval-trackers", help="compare tracker algorithms (SORT / DeepSORT / ByteTrack / BoT-SORT)")
    eval_trackers.add_argument("--methods", nargs="+", default=list(DEFAULT_TRACKER_METHODS))
    eval_trackers.add_argument("--dist-threshold", type=float, default=15.0)
    eval_trackers.add_argument("--max-clips", type=int, default=0)
    eval_trackers.add_argument("--device", default="")
    eval_trackers.add_argument("--yolo-variant", choices=list(YOLO_VARIANTS), default=DEFAULT_YOLO_VARIANT)
    eval_trackers.add_argument("--yolo-model-path", default="")

    eval_ablation = sub.add_parser("eval-ablation", help="compare tracker ablations on the test split")
    eval_ablation.add_argument("--modes", nargs="+", default=[])
    eval_ablation.add_argument("--dist-threshold", type=float, default=15.0)
    eval_ablation.add_argument("--max-clips", type=int, default=0)
    eval_ablation.add_argument("--device", default="")
    eval_ablation.add_argument("--yolo-variant", choices=list(YOLO_VARIANTS), default=DEFAULT_YOLO_VARIANT)
    eval_ablation.add_argument("--yolo-model-path", default="")

    sub.add_parser("report", help="generate the markdown report from methods.json and ablation.json")

    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_layout()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        from .dataset import prepare_command

        prepare_command(args)
    elif args.command == "train-yolo":
        from .yolo import train_yolo_command

        train_yolo_command(args)
    elif args.command == "cache-yolo":
        from .yolo import cache_yolo_command

        cache_yolo_command(args)
    elif args.command == "train-tracknet":
        from .tracknet import train_tracknet_command

        train_tracknet_command(args)
    elif args.command == "train-tracknet-v2":
        from .tracknet_v2 import train_tracknet_v2_command

        train_tracknet_v2_command(args)
    elif args.command == "train-tracknet-v3":
        from .tracknet_v3 import train_tracknet_v3_command

        train_tracknet_v3_command(args)
    elif args.command == "train-tracknet-v4":
        from .tracknet_v4 import train_tracknet_v4_command

        train_tracknet_v4_command(args)
    elif args.command == "train-tracknet-v5":
        from .tracknet_v5 import train_tracknet_v5_command

        train_tracknet_v5_command(args)
    elif args.command == "eval-methods":
        from .yolo import eval_methods_command

        eval_methods_command(args)
    elif args.command == "eval-trackers":
        from .tracker_comparison import tracker_comparison_command

        tracker_comparison_command(args)
    elif args.command == "eval-ablation":
        from .yolo import eval_ablation_command

        eval_ablation_command(args)
    elif args.command == "report":
        from .report import report_command

        report_command(args)
    else:
        parser.error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
