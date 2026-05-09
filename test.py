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

from dataset import REGRESSION_TASK, MPDDElderDataset, collate_batch, resolve_project_path
from device_utils import build_model_on_available_device
from models import TorchcatBaseline


PROJECT_ROOT = Path(__file__).resolve().parent
SUBTRACK_LOG_DIRS = {
    "A-V-P": "A-V-P",
    "A-V-G-P": "A-V-G-P",
    "G-P": "G-P",
}
PATH_ANCHORS = (
    "runs",
    "MPDD-AVG2026-test",
    "MPDD-AVG2026-trainval",
    "MPDD-AVG2026-raw",
    "make_submission_forcodabench",
    "checkpoints",
    "logs",
    "predictions",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(resolve_project_path(config_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference from a trained MPDD-AVG baseline checkpoint.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth file or checkpoint directory.")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--personality_npy", default="")
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--logs_dir", default=defaults["logs_dir"])
    parser.add_argument("--sample_csv", default="", help="Optional CodaBench sample CSV with an id column.")
    parser.add_argument(
        "--prediction_csv",
        default="",
        help="Optional output path for binary.csv or ternary.csv. If empty, will write under logs_dir.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default="config.json")
    known_args, _ = base_parser.parse_known_args()
    defaults = load_config(known_args.config)
    parser = build_parser(defaults)
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(f"mpdd_avg_test_{time.time_ns()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    return logger


def remap_repo_path(path_like: str | Path) -> str:
    path = Path(path_like)
    if path.exists():
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(resolved_path)

    if path.is_absolute():
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)

    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        candidate = PROJECT_ROOT.joinpath(*path.parts[1:])
        if candidate.exists():
            return candidate.relative_to(PROJECT_ROOT).as_posix()

    if path.parts and path.parts[0] in PATH_ANCHORS:
        candidate = PROJECT_ROOT.joinpath(*path.parts)
        return candidate.relative_to(PROJECT_ROOT).as_posix()

    for anchor in PATH_ANCHORS:
        if anchor not in path.parts:
            continue
        anchor_index = path.parts.index(anchor)
        candidate = PROJECT_ROOT.joinpath(*path.parts[anchor_index:])
        if candidate.exists() or anchor in {"logs", "predictions", "runs"}:
            return candidate.relative_to(PROJECT_ROOT).as_posix()

    return path.as_posix()


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


def resolve_checkpoint_path(path_like: str | Path) -> Path:
    checkpoint_path = resolve_project_path(remap_repo_path(path_like))
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    stable_checkpoint = checkpoint_path / "best_model.pth"
    if stable_checkpoint.is_file():
        return stable_checkpoint

    for pattern in ("best_model_*.pth", "last_model_*.pth", "*.pth"):
        candidates = sorted(checkpoint_path.glob(pattern))
        if candidates:
            return max(candidates, key=lambda item: (item.stat().st_mtime, item.name))
    for pattern in ("best_model.pth", "best_model_*.pth", "last_model_*.pth", "*.pth"):
        candidates = sorted(checkpoint_path.rglob(pattern))
        if candidates:
            return max(candidates, key=lambda item: (item.stat().st_mtime, item.name))
    raise FileNotFoundError(f"No checkpoint .pth file found under: {checkpoint_path}")


def resolve_test_data_root(args_data_root: str, checkpoint_data_root: str) -> str:
    if args_data_root:
        return remap_repo_path(args_data_root)

    checkpoint_root = remap_repo_path(checkpoint_data_root)
    if "MPDD-AVG2026-trainval" in checkpoint_root:
        candidate = checkpoint_root.replace("MPDD-AVG2026-trainval", "MPDD-AVG2026-test")
        if resolve_project_path(candidate).exists():
            return candidate
    return checkpoint_root


def read_sample_ids(sample_csv: str | Path) -> list[int]:
    csv_path = resolve_project_path(remap_repo_path(sample_csv))
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Sample CSV is empty: {csv_path}")
    id_column = "id" if "id" in rows[0] else "ID" if "ID" in rows[0] else ""
    if not id_column:
        raise ValueError(f"Sample CSV must contain an id column: {csv_path}")
    return [int(float(row[id_column])) for row in rows if str(row.get(id_column, "")).strip()]


def infer_test_ids_from_data_root(data_root: str | Path) -> list[int]:
    root = resolve_project_path(remap_repo_path(data_root))
    if not root.exists():
        raise FileNotFoundError(f"Cannot infer test IDs because data_root does not exist: {root}")

    ids: set[int] = set()
    for path in root.rglob("*"):
        if path.is_dir() and path.name.isdigit():
            ids.add(int(path.name))
        elif path.is_file() and path.suffix.lower() == ".npy" and path.stem.isdigit():
            ids.add(int(path.stem))
    if not ids:
        raise RuntimeError(f"No numeric test IDs found under: {root}")
    return sorted(ids)


def build_unlabeled_task_maps(sample_ids: list[int]) -> dict[str, Any]:
    return {
        "test_map": {sample_id: 0 for sample_id in sample_ids},
        "source_split_map": {sample_id: "test" for sample_id in sample_ids},
    }


def load_test_task_maps(data_root: str | Path, sample_csv: str | Path = "") -> dict[str, Any]:
    sample_ids = read_sample_ids(sample_csv) if sample_csv else infer_test_ids_from_data_root(data_root)
    return build_unlabeled_task_maps(sample_ids)


def inverse_normalize_phq(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        return 0.0
    if value >= math.log1p(27.0):
        return 27.0
    return float(math.expm1(value))


def write_prediction_csv_from_logits(
    ids: list[int],
    class_pred: list[int],
    phq_pred: list[float] | None,
    task: str,
    output_path: str | Path,
) -> str:
    output_path = resolve_project_path(remap_repo_path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if task == "binary":
        pred_column = "binary_pred"
    elif task == "ternary":
        pred_column = "ternary_pred"
    else:
        raise ValueError(f"Prediction CSV is only supported for binary/ternary tasks, got task={task}")

    phq_values = phq_pred if phq_pred is not None and len(phq_pred) == len(ids) else [0.0] * len(ids)
    rows = []
    for index, sample_id in enumerate(ids):
        rows.append(
            {
                "id": int(sample_id),
                pred_column: int(class_pred[index]),
                "phq9_pred": f"{inverse_normalize_phq(float(phq_values[index])):.6f}",
            }
        )

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", pred_column, "phq9_pred"])
        writer.writeheader()
        writer.writerows(rows)
    return to_project_relative_path(output_path)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, use_regression_head: bool) -> dict[str, Any]:
    model.eval()
    all_ids: list[int] = []
    all_class_pred: list[int] = []
    all_phq_pred: list[float] = []

    for batch in loader:
        outputs = model(
            audio=batch["audio"].to(device) if "audio" in batch else None,
            video=batch["video"].to(device) if "video" in batch else None,
            gait=batch["gait"].to(device) if "gait" in batch else None,
            personality=batch["personality"].to(device),
            pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
        )
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("class_logits"))
            phq = outputs.get("phq_pred", outputs.get("regression"))
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            phq = outputs[1] if use_regression_head and len(outputs) > 1 else None
        else:
            logits = outputs
            phq = None

        if logits is None:
            raise RuntimeError("Model output missing logits; cannot produce class predictions.")

        all_ids.extend(int(item) for item in batch["pid"].cpu().tolist())
        all_class_pred.extend(int(item) for item in torch.argmax(logits, dim=-1).detach().cpu().tolist())
        if use_regression_head and phq is not None:
            all_phq_pred.extend(float(item) for item in phq.detach().float().cpu().view(-1).tolist())

    return {
        "ids": all_ids,
        "class_pred": all_class_pred,
        "phq_pred": all_phq_pred if len(all_phq_pred) == len(all_ids) else None,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    track = require_checkpoint_value(checkpoint, "track")
    task = require_checkpoint_value(checkpoint, "task")
    regression_label = checkpoint.get("regression_label", "")
    subtrack = require_checkpoint_value(checkpoint, "subtrack")
    encoder_type = require_checkpoint_value(checkpoint, "encoder_type")
    audio_feature = require_checkpoint_value(checkpoint, "audio_feature")
    video_feature = require_checkpoint_value(checkpoint, "video_feature")
    data_root = resolve_test_data_root(args.data_root, require_checkpoint_value(checkpoint, "data_root"))
    personality_npy = remap_repo_path(args.personality_npy or require_checkpoint_value(checkpoint, "personality_npy"))
    target_t = int(require_checkpoint_value(checkpoint, "target_t"))
    experiment_name = checkpoint.get("experiment_name", checkpoint_path.parent.name)

    timestamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    logs_root = resolve_project_path(remap_repo_path(args.logs_dir))
    log_dir = resolve_track_task_dir(logs_root, track, subtrack, task, experiment_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()
    logger.info("Checkpoint: %s", to_project_relative_path(checkpoint_path))
    logger.info("Data root: %s", data_root)
    logger.info("Personality file: %s", personality_npy)

    task_maps = load_test_task_maps(data_root=data_root, sample_csv=args.sample_csv)
    test_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack=subtrack,
        task=task,
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality_npy,
        phq_map=None,
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
    pred_payload = predict(model, test_loader, device, use_regression_head=use_regression_head)

    prediction_csv_path = args.prediction_csv
    if not prediction_csv_path and task in {"binary", "ternary"}:
        prediction_csv_path = log_dir / f"{task}_{checkpoint_path.stem}.csv"
    if not prediction_csv_path:
        raise ValueError("prediction_csv is required for non-classification tasks.")

    prediction_csv_rel = write_prediction_csv_from_logits(
        ids=pred_payload["ids"],
        class_pred=pred_payload["class_pred"],
        phq_pred=pred_payload.get("phq_pred"),
        task=task,
        output_path=prediction_csv_path,
    )

    result_payload = {
        "checkpoint": to_project_relative_path(checkpoint_path),
        "track": track,
        "task": task,
        "subtrack": subtrack,
        "encoder_type": encoder_type,
        "audio_feature": audio_feature,
        "video_feature": video_feature,
        "regression_label": regression_label if task == REGRESSION_TASK else "",
        "data_root": data_root,
        "predictions_path": prediction_csv_rel,
        "prediction_count": len(pred_payload["ids"]),
    }
    result_path = log_dir / f"prediction_result_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    logger.info("Prediction CSV saved to: %s", prediction_csv_rel)
    logger.info("Prediction count: %d", len(pred_payload["ids"]))
    logger.info("Prediction metadata saved to: %s", to_project_relative_path(result_path))


if __name__ == "__main__":
    main()
