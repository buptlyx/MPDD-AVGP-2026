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
from metrics import build_score_weights, evaluate_model
from models import TorchcatBaseline


PROJECT_ROOT = Path(__file__).resolve().parent
SUBTRACK_LOG_DIRS = {
    "A-V-P": "A-V-P",
    "A-V-G-P": "A-V-G-P",
    "G-P": "G-P",
}
METRIC_ARRAY_KEYS = {"ids", "y_true", "y_pred", "class_true", "class_pred", "phq_true", "phq_pred"}
ABLATION_AUTO_SUBTRACKS = {
    "G-P": ("gait", "personality"),
}
SUMMARY_METRICS = (
    ("ScoreTrack", "scoretrack"),
    ("Macro-F1", "f1"),
    ("ACC", "acc"),
    ("Kappa", "kappa"),
    ("CCC", "ccc"),
    ("RMSE", "rmse"),
    ("MAE", "mae"),
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(resolve_project_path(config_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference from a trained MPDD-AVG baseline checkpoint.")
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
    parser.add_argument(
        "--prediction_csv",
        default="",
        help="Optional output path for binary.csv or ternary.csv. If empty, will write under logs_dir.",
    )
    # NOTE: 移除打分相关参数（score_alpha/beta/gamma）与 ablation
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
    # 仅用于 Dataset 初始化占位，不用于评估
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
        # 如果你给了带标签的 split_csv，这里仍然能加载标签，
        # 但本脚本不再使用标签做评估，只用于保证 ID 列表一致。
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
            pass

    anchors = (
        "MPDD-AVG2026-test",
        "MPDD-AVG2026-trainval",
        "MPDD-AVG2026-raw",
        "make_submission_forcodabench",
        "checkpoints",
        "logs",
        "predictions",
    )
    if not path.is_absolute():
        if path.parts and path.parts[0] == PROJECT_ROOT.name:
            candidate = PROJECT_ROOT.joinpath(*path.parts[1:])
            if candidate.exists():
                return candidate.relative_to(PROJECT_ROOT).as_posix()
        if not path.parts or path.parts[0] not in anchors:
            return path.as_posix()

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

    rows = []
    for i, sample_id in enumerate(ids):
        row = {"id": int(sample_id), pred_column: int(class_pred[i])}
        if phq_pred is not None:
            row["phq9_pred"] = f"{inverse_normalize_phq(float(phq_pred[i])):.6f}"
        rows.append(row)

    fieldnames = ["id", pred_column] + (["phq9_pred"] if phq_pred is not None else [])
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
        # collate_batch 通常会返回 dict
        batch_ids = batch.get("ids", batch.get("id"))
        if batch_ids is None:
            raise KeyError("Batch missing 'ids' field. Please check collate_batch output.")

        # ids 可能是 Tensor / list
        if torch.is_tensor(batch_ids):
            batch_ids_list = [int(x) for x in batch_ids.cpu().tolist()]
        else:
            batch_ids_list = [int(x) for x in batch_ids]

        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        outputs = model(batch, only_modality=None) if callable(getattr(model, "__call__", None)) else model(**batch)

        # 兼容 TorchcatBaseline 常见返回：可能是 dict 或 tuple
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("class_logits"))
            phq = outputs.get("phq_pred", outputs.get("regression"))
        else:
            # 如果模型 forward 返回 (logits, phq) 或仅 logits
            logits = outputs[0]
            phq = outputs[1] if (use_regression_head and len(outputs) > 1) else None

        if logits is None:
            raise RuntimeError("Model output missing logits; cannot produce class predictions.")

        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()

        all_ids.extend(batch_ids_list)
        all_class_pred.extend([int(p) for p in preds])

        if use_regression_head and phq is not None:
            phq_vals = phq.detach().float().cpu().view(-1).tolist()
            all_phq_pred.extend([float(v) for v in phq_vals])

    payload: dict[str, Any] = {"ids": all_ids, "class_pred": all_class_pred}
    if use_regression_head:
        payload["phq_pred"] = all_phq_pred if len(all_phq_pred) == len(all_ids) else None
    else:
        payload["phq_pred"] = None
    return payload


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
        label_map=task_maps["test_map"],  # 占位，不用于评估
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

    pred_payload = predict(model, test_loader, device, use_regression_head=use_regression_head)

    prediction_csv_path = args.prediction_csv
    if not prediction_csv_path and task in {"binary", "ternary"}:
        prediction_csv_path = log_dir / f"{task}_{checkpoint_path.stem}.csv"
    if not prediction_csv_path:
        raise ValueError("prediction_csv is required (or task must be binary/ternary to auto-generate a path).")

    prediction_csv_rel = write_prediction_csv_from_logits(
        ids=pred_payload["ids"],
        class_pred=pred_payload["class_pred"],
        phq_pred=pred_payload.get("phq_pred"),
        task=task,
        output_path=prediction_csv_path,
    )
    logger.info("Prediction CSV saved to: %s", prediction_csv_rel)


if __name__ == "__main__":
    main()