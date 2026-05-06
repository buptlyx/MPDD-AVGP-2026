from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import REGRESSION_TASK, MPDDElderDataset, collate_batch, load_task_maps, resolve_project_path
from device_utils import build_model_on_available_device
from metrics import evaluate_model
from models import TorchcatBaseline


PROJECT_ROOT = Path(__file__).resolve().parent
SUBTRACK_LOG_DIRS = {
    "A-V+P": "A-V-P",
    "A-V-G+P": "A-V-G+P",
    "G+P": "G-P",
}
METRIC_ARRAY_KEYS = {"ids", "y_true", "y_pred", "class_true", "class_pred", "phq_true", "phq_pred"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(resolve_project_path(config_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained MPDD-AVG baseline checkpoint.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--split_csv", default="")
    parser.add_argument("--personality_npy", default="")
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--logs_dir", default=defaults["logs_dir"])
    parser.add_argument("--sample_csv", default="", help="Optional CodaBench sample CSV with an id column.")
    parser.add_argument("--prediction_csv", default="", help="Optional output path for binary.csv or ternary.csv.")
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default="config.json")
    known_args, _ = base_parser.parse_known_args()
    defaults = load_config(known_args.config)
    parser = build_parser(defaults)
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(f"elder_track1_test_{time.time_ns()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    return logger


def append_summary_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_track_task_dir(root: Path, track: str, subtrack: str, task: str, experiment_name: str) -> Path:
    subtrack_dir = SUBTRACK_LOG_DIRS.get(subtrack, subtrack.replace("+", "-"))
    return root / track / subtrack_dir / task / experiment_name


def to_project_relative_path(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def require_checkpoint_value(checkpoint: dict[str, Any], key: str) -> Any:
    value = checkpoint.get(key)
    if value in (None, ""):
        raise KeyError(f"Checkpoint missing required field: {key}")
    return value


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in METRIC_ARRAY_KEYS}


def read_sample_ids(sample_csv: str | Path) -> list[int]:
    csv_path = resolve_project_path(remap_repo_path(sample_csv))
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "id" not in rows[0]:
        raise ValueError(f"Sample CSV must contain an id column: {csv_path}")
    return [int(row["id"]) for row in rows]


def infer_test_ids_from_data_root(data_root: str | Path) -> list[int]:
    root = resolve_project_path(remap_repo_path(data_root))
    if not root.exists():
        raise FileNotFoundError(f"Cannot infer test IDs because data_root does not exist: {root}")

    ids: set[int] = set()
    for path in root.rglob("*"):
        if path.is_dir() and path.name.isdigit():
            ids.add(int(path.name))
    if not ids:
        raise RuntimeError(f"No numeric sample ID directories found under: {root}")
    return sorted(ids)


def build_unlabeled_task_maps(sample_ids: list[int]) -> dict[str, Any]:
    label_map = {sample_id: 0 for sample_id in sample_ids}
    return {
        "test_map": label_map,
        "source_split_map": {sample_id: "test" for sample_id in sample_ids},
        "test_phq_map": {sample_id: 0.0 for sample_id in sample_ids},
    }


def load_test_task_maps(
    split_csv: str | Path,
    task: str,
    regression_label: str,
    data_root: str | Path,
    sample_csv: str | Path = "",
) -> dict[str, Any]:
    split_path = resolve_project_path(remap_repo_path(split_csv)) if split_csv else None
    if split_path is not None and split_path.exists():
        return load_task_maps(split_path, task, regression_label)

    sample_ids = read_sample_ids(sample_csv) if sample_csv else infer_test_ids_from_data_root(data_root)
    return build_unlabeled_task_maps(sample_ids)


def inverse_normalize_phq(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        return 0.0
    if value >= math.log1p(27.0):
        return 27.0
    return float(math.expm1(value))


def write_prediction_csv(metrics: dict[str, Any], task: str, output_path: str | Path) -> str:
    output_path = resolve_project_path(remap_repo_path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if task == "binary":
        pred_column = "binary_pred"
    elif task == "ternary":
        pred_column = "ternary_pred"
    else:
        raise ValueError(f"Prediction CSV is only supported for binary/ternary tasks, got task={task}")

    class_preds = metrics.get("class_pred", metrics.get("y_pred", []))
    phq_preds = metrics.get("phq_pred", metrics.get("y_pred", []))
    rows = []
    for sample_id, class_pred, phq_pred in zip(metrics["ids"], class_preds, phq_preds):
        rows.append(
            {
                "id": int(sample_id),
                pred_column: int(class_pred),
                "phq9_pred": f"{inverse_normalize_phq(float(phq_pred)):.6f}",
            }
        )

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", pred_column, "phq9_pred"])
        writer.writeheader()
        writer.writerows(rows)
    return to_project_relative_path(output_path)


def remap_repo_path(path_like: str | Path) -> str:
    path = Path(path_like)
    if path.exists():
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(resolved_path)

    anchors = (
        "MPDD-AVG2026-test",
        "MPDD-AVG2026-trainval",
        "MPDD-AVG2026-raw",
        "make_submission_forcodabench",
        "checkpoints",
        "logs",
        "predictions",
    )
    for anchor in anchors:
        if anchor not in path.parts:
            continue
        anchor_index = path.parts.index(anchor)
        candidate = PROJECT_ROOT.joinpath(*path.parts[anchor_index:])
        if candidate.exists() or anchor in {"logs", "predictions"}:
            return candidate.relative_to(PROJECT_ROOT).as_posix()

    if not path.is_absolute() and path.parts and path.parts[0] == PROJECT_ROOT.name:
        candidate = PROJECT_ROOT.joinpath(*path.parts[1:])
        if candidate.exists():
            return candidate.relative_to(PROJECT_ROOT).as_posix()

    if not path.is_absolute():
        return path.as_posix()
    return str(path)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_project_path(remap_repo_path(args.checkpoint))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    track = require_checkpoint_value(checkpoint, "track")
    task = require_checkpoint_value(checkpoint, "task")
    regression_label = checkpoint.get("regression_label", "")
    subtrack = require_checkpoint_value(checkpoint, "subtrack")
    encoder_type = require_checkpoint_value(checkpoint, "encoder_type")
    audio_feature = require_checkpoint_value(checkpoint, "audio_feature")
    video_feature = require_checkpoint_value(checkpoint, "video_feature")
    data_root = remap_repo_path(args.data_root or require_checkpoint_value(checkpoint, "data_root"))
    split_csv = remap_repo_path(args.split_csv or require_checkpoint_value(checkpoint, "split_csv"))
    personality_npy = remap_repo_path(args.personality_npy or require_checkpoint_value(checkpoint, "personality_npy"))
    target_t = int(require_checkpoint_value(checkpoint, "target_t"))
    experiment_name = checkpoint.get("experiment_name", checkpoint_path.parent.name)

    timestamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    logs_root = resolve_project_path(remap_repo_path(args.logs_dir))
    log_dir = resolve_track_task_dir(logs_root, track, subtrack, task, experiment_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()

    task_maps = load_test_task_maps(
        split_csv=split_csv,
        task=task,
        regression_label=regression_label or "label2",
        data_root=data_root,
        sample_csv=args.sample_csv,
    )
    test_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack=subtrack,
        task=task,
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality_npy,
        phq_map=task_maps.get("test_phq_map"),
        target_t=target_t,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    model_kwargs = dict(require_checkpoint_value(checkpoint, "model_kwargs"))
    model, device = build_model_on_available_device(
        lambda: TorchcatBaseline(**model_kwargs),
        args.device,
        logger,
    )
    model.load_state_dict(require_checkpoint_value(checkpoint, "model_state"))
    use_regression_head = bool(model_kwargs.get("use_regression_head", False))
    is_regression_task = task == REGRESSION_TASK
    criterion = (nn.CrossEntropyLoss(), nn.MSELoss()) if use_regression_head else nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, criterion, device, task)
    metric_summary = summarize_metrics(metrics)
    checkpoint_rel = to_project_relative_path(checkpoint_path)
    prediction_csv_rel = ""
    prediction_csv_path = args.prediction_csv
    if not prediction_csv_path and task in {"binary", "ternary"}:
        prediction_csv_path = log_dir / f"{task}_{checkpoint_path.stem}.csv"
    if prediction_csv_path:
        prediction_csv_rel = write_prediction_csv(metrics, task, prediction_csv_path)

    result_payload = {
        "checkpoint": checkpoint_rel,
        "track": track,
        "task": task,
        "subtrack": subtrack,
        "encoder_type": encoder_type,
        "audio_feature": audio_feature,
        "video_feature": video_feature,
        "regression_label": regression_label if is_regression_task else "",
        "metrics": metric_summary,
        "predictions_path": prediction_csv_rel,
    }
    result_path = log_dir / f"test_result_only_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    summary_row = {
        "timestamp": timestamp,
        "mode": "test_only",
        "track": track,
        "task": task,
        "subtrack": subtrack,
        "encoder_type": encoder_type,
        "audio_feature": audio_feature,
        "video_feature": video_feature,
        "checkpoint": checkpoint_rel,
        "predictions_path": prediction_csv_rel,
        "Macro-F1": f"{metrics.get('f1', 0.0):.6f}",
        "ACC": f"{metrics.get('acc', 0.0):.6f}",
        "Kappa": f"{metrics.get('kappa', 0.0):.6f}",
        "CCC": f"{metrics['ccc']:.6f}",
        "RMSE": f"{metrics['rmse']:.6f}",
        "MAE": f"{metrics['mae']:.6f}",
        "R2": f"{metrics.get('r2', ''):.6f}" if is_regression_task else "",
    }
    if is_regression_task:
        summary_row["regression_label"] = regression_label
    append_summary_row(log_dir / f"{experiment_name}_test_only.csv", summary_row)
    if prediction_csv_rel:
        logger.info("Prediction CSV saved to: %s", prediction_csv_rel)
    logger.info("Test-only metrics saved to: %s", to_project_relative_path(result_path))


if __name__ == "__main__":
    main()
