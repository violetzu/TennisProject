"""CLI entry point for courttest."""
from __future__ import annotations

import argparse
import sys

from .paths import CHECKPOINTS_DIR, DL_WEIGHTS, RESULTS_DIR, WEIGHTS, YOLO_TRAINED_VARIANTS, _yolo_run_best, ensure_layout
from .train_yolo import YOLO_VARIANTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Courttest: court keypoint detection evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train-yolo ─────────────────────────────────────────────────────────
    ty = sub.add_parser("train-yolo", help="train a YOLO-pose variant on court dataset")
    ty.add_argument("--variant", choices=list(YOLO_VARIANTS), required=True)
    ty.add_argument("--imgsz", type=int, default=640)
    ty.add_argument("--epochs", type=int, default=500)
    ty.add_argument("--patience", type=int, default=30)
    ty.add_argument("--batch", type=int, default=48)
    ty.add_argument("--device", default="")

    # ── train-heatmap ──────────────────────────────────────────────────────
    th = sub.add_parser("train-heatmap", help="train HeatmapNet (MobileNetV3 encoder-decoder)")
    th.add_argument("--epochs", type=int, default=500)
    th.add_argument("--batch-size", type=int, default=8)
    th.add_argument("--lr", type=float, default=1e-3)
    th.add_argument("--patience", type=int, default=20)
    th.add_argument("--device", default="")

    # ── train-regressor ────────────────────────────────────────────────────
    tr = sub.add_parser("train-regressor", help="train ResNet50 regression model")
    tr.add_argument("--epochs", type=int, default=500)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--patience", type=int, default=20)
    tr.add_argument("--device", default="")

    # ── eval-all ───────────────────────────────────────────────────────────
    ea = sub.add_parser("eval-all", help="evaluate all methods + Hough baseline")
    ea.add_argument("--device", default="")
    ea.add_argument("--no-hough", action="store_true")
    ea.add_argument("--per-kp", action="store_true")

    # ── eval-yolo ──────────────────────────────────────────────────────────
    ey = sub.add_parser("eval-yolo", help="evaluate one YOLO variant")
    ey.add_argument("--variant", choices=list(WEIGHTS), default="yolo26n_640_250ep")
    ey.add_argument("--imgsz", type=int, default=None)
    ey.add_argument("--device", default="")
    ey.add_argument("--per-kp", action="store_true")

    # ── eval-hough ─────────────────────────────────────────────────────────
    sub.add_parser("eval-hough", help="evaluate Hough baseline")

    # ── eval-heatmap ───────────────────────────────────────────────────────
    eh = sub.add_parser("eval-heatmap", help="evaluate HeatmapNet")
    eh.add_argument("--weights", default="", help="path to .pt (default: artifacts/checkpoints)")
    eh.add_argument("--device", default="")
    eh.add_argument("--per-kp", action="store_true")

    # ── eval-regressor ─────────────────────────────────────────────────────
    er = sub.add_parser("eval-regressor", help="evaluate ResNet50 regressor")
    er.add_argument("--weights", default="")
    er.add_argument("--device", default="")
    er.add_argument("--per-kp", action="store_true")

    return parser


def run(argv=None) -> None:
    ensure_layout()
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── training commands ──────────────────────────────────────────────────
    if args.command == "train-yolo":
        from .train_yolo import train_yolo_court
        train_yolo_court(
            variant=args.variant,
            imgsz=args.imgsz,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            device=args.device,
        )

    elif args.command == "train-heatmap":
        from .model_heatmap import train_heatmap
        from .paths import COURT_DATASET
        train_heatmap(
            dataset_root=COURT_DATASET,
            save_path=DL_WEIGHTS["heatmap_mobilenet"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device=args.device,
        )

    elif args.command == "train-regressor":
        from .model_regressor import train_regressor
        from .paths import COURT_DATASET
        train_regressor(
            dataset_root=COURT_DATASET,
            save_path=DL_WEIGHTS["resnet50_regressor"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device=args.device,
        )

    # ── eval commands ──────────────────────────────────────────────────────
    elif args.command == "eval-hough":
        from .baselines import evaluate_hough
        from .report import print_table, save_json
        r = evaluate_hough()
        print_table([r])
        save_json([r], RESULTS_DIR / "hough.json")

    elif args.command == "eval-yolo":
        from .yolo_eval import evaluate_yolo
        from .report import print_per_kp_table, print_table, save_json
        r = evaluate_yolo(args.variant, imgsz=args.imgsz, device=args.device)
        print_table([r])
        if args.per_kp:
            print_per_kp_table(r)
        save_json([r], RESULTS_DIR / f"{args.variant}.json")

    elif args.command == "eval-heatmap":
        from .dl_eval import evaluate_heatmap
        from .report import print_per_kp_table, print_table, save_json
        wp = args.weights or str(DL_WEIGHTS["heatmap_mobilenet"])
        r = evaluate_heatmap(weights_path=__import__("pathlib").Path(wp), device=args.device)
        print_table([r])
        if args.per_kp:
            print_per_kp_table(r)
        save_json([r], RESULTS_DIR / "heatmap_mobilenet.json")

    elif args.command == "eval-regressor":
        from .dl_eval import evaluate_regressor
        from .report import print_per_kp_table, print_table, save_json
        wp = args.weights or str(DL_WEIGHTS["resnet50_regressor"])
        r = evaluate_regressor(weights_path=__import__("pathlib").Path(wp), device=args.device)
        print_table([r])
        if args.per_kp:
            print_per_kp_table(r)
        save_json([r], RESULTS_DIR / "resnet50_regressor.json")

    elif args.command == "eval-all":
        from .baselines import evaluate_hough
        from .dl_eval import evaluate_heatmap, evaluate_regressor
        from .report import print_per_kp_table, print_table, save_json
        from .yolo_eval import evaluate_yolo

        results = []

        # Pre-existing YOLO variants
        for variant in WEIGHTS:
            print(f"\n[courttest] evaluating {variant} ...")
            try:
                r = evaluate_yolo(variant, device=args.device)
                results.append(r)
                print(f"  → mPCK@10={r.mpck_10*100:.1f}%  FPS={r.fps:.1f}")
            except Exception as e:
                print(f"  [SKIP] {variant}: {e}", file=sys.stderr)

        # Newly trained YOLO variants
        for variant in YOLO_TRAINED_VARIANTS:
            wp = _yolo_run_best(variant)
            if not wp.exists():
                print(f"\n[SKIP] {variant}: not trained yet")
                continue
            print(f"\n[courttest] evaluating {variant} ...")
            try:
                from .yolo_eval import evaluate_yolo_custom
                r = evaluate_yolo_custom(variant, wp, device=args.device)
                results.append(r)
                print(f"  → mPCK@10={r.mpck_10*100:.1f}%  FPS={r.fps:.1f}")
            except Exception as e:
                print(f"  [SKIP] {variant}: {e}", file=sys.stderr)

        # HeatmapNet
        if DL_WEIGHTS["heatmap_mobilenet"].exists():
            print("\n[courttest] evaluating heatmap_mobilenet ...")
            try:
                r = evaluate_heatmap(DL_WEIGHTS["heatmap_mobilenet"], device=args.device)
                results.append(r)
                print(f"  → mPCK@10={r.mpck_10*100:.1f}%  FPS={r.fps:.1f}")
            except Exception as e:
                print(f"  [SKIP] heatmap_mobilenet: {e}", file=sys.stderr)
        else:
            print("\n[SKIP] heatmap_mobilenet: not trained yet (run train-heatmap first)")

        # ResNet50
        if DL_WEIGHTS["resnet50_regressor"].exists():
            print("\n[courttest] evaluating resnet50_regressor ...")
            try:
                r = evaluate_regressor(DL_WEIGHTS["resnet50_regressor"], device=args.device)
                results.append(r)
                print(f"  → mPCK@10={r.mpck_10*100:.1f}%  FPS={r.fps:.1f}")
            except Exception as e:
                print(f"  [SKIP] resnet50_regressor: {e}", file=sys.stderr)
        else:
            print("\n[SKIP] resnet50_regressor: not trained yet (run train-regressor first)")

        # Hough
        if not args.no_hough:
            print("\n[courttest] evaluating hough_baseline ...")
            r_h = evaluate_hough()
            results.append(r_h)
            print(f"  → mPCK@10={r_h.mpck_10*100:.1f}%  FPS={r_h.fps:.1f}")

        print("\n")
        print_table(results)
        if args.per_kp:
            for r in results:
                print_per_kp_table(r)

        save_json(results, RESULTS_DIR / "eval_all.json")
