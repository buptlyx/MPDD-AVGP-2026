from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBTRACK_ALIASES = {
    "A-V-P": "A-V-P",
    "A-V-P": "A-V-P",
    "A-V-G-P": "A-V-G-P",
    "G-P": "G-P",
    "G-P": "G-P",
}
SUBTRACK_TRAIN_NAMES = {
    "A-V-P": "A-V-P",
    "A-V-G-P": "A-V-G-P",
    "G-P": "G-P",
}
SUBTRACK_TAGS = {
    "A-V-P": "avp",
    "A-V-G-P": "avgp",
    "G-P": "gp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run binary and ternary training for one MPDD-AVG track/subtrack, "
            "evaluate the new checkpoints, and package the outputs."
        )
    )
    parser.add_argument("--track", default="Track1", choices=["Track1", "Track2"])
    parser.add_argument(
        "--subtrack",
        default="A-V-G-P",
        choices=["A-V-P", "A-V-P", "A-V-G-P", "G-P", "G-P"],
        help="Subtrack name. A-V-P/G-P aliases are accepted.",
    )
    parser.add_argument("--run_id", default="", help="Optional run id used in experiment names and output folders.")
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", "python"),
        help="Python interpreter passed to train/test scripts.",
    )
    parser.add_argument("--bash-bin", default="bash", help="Bash executable used to run existing .sh scripts.")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--checkpoints-dir", default=os.environ.get("CHECKPOINTS_DIR", "checkpoints"))
    parser.add_argument("--train-logs-dir", default=os.environ.get("TRAIN_LOGS_DIR", "logs/train"))
    parser.add_argument("--test-logs-dir", default=os.environ.get("TEST_LOGS_DIR", "logs/test"))
    parser.add_argument("--output-dir", default="runs", help="Root directory for per-run outputs.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and test existing checkpoints.")
    parser.add_argument("--skip-test", action="store_true", help="Skip test and packaging steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands without running them.")
    parser.add_argument("--archive-artifacts", action="store_true", help="Also zip checkpoints, logs, and predictions.")
    parser.add_argument("--keep-going", action="store_true", help="Continue with ternary if binary fails, then fail at the end.")
    return parser.parse_args()


def normalize_subtrack(value: str) -> str:
    try:
        return SUBTRACK_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported subtrack: {value}") from exc


def task_experiment_name(track: str, subtrack_dir: str, task: str, run_id: str) -> str:
    track_tag = track.lower()
    subtrack_tag = SUBTRACK_TAGS[subtrack_dir]
    return f"{track_tag}_{subtrack_tag}_{task}_{run_id}"


def run_command(command: list[str], env: dict[str, str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def normalize_shell_line_endings(paths: list[Path]) -> None:
    for path in sorted(set(paths)):
        data = path.read_bytes()
        normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if normalized != data:
            path.write_bytes(normalized)


def rel_posix(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def add_directory_to_zip(zip_file: zipfile.ZipFile, source: Path, arc_prefix: Path) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file():
            zip_file.write(path, arc_prefix / path.relative_to(source))


def write_artifact_archive(
    archive_path: Path,
    manifest_path: Path,
    run_root: Path,
    checkpoints: dict[str, Path],
    train_logs: dict[str, Path],
    test_logs: dict[str, Path],
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname="manifest.json")
        add_directory_to_zip(zf, run_root / "predictions", Path("predictions"))
        for task, path in checkpoints.items():
            add_directory_to_zip(zf, path, Path("checkpoints") / task)
        for task, path in train_logs.items():
            add_directory_to_zip(zf, path, Path("logs") / "train" / task)
        for task, path in test_logs.items():
            add_directory_to_zip(zf, path, Path("logs") / "test" / task)


def main() -> None:
    args = parse_args()
    subtrack_dir = normalize_subtrack(args.subtrack)
    subtrack_train_name = SUBTRACK_TRAIN_NAMES[subtrack_dir]
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_root = (PROJECT_ROOT / args.output_dir / f"{args.track}_{subtrack_dir}_{run_id}").resolve()
    predictions_dir = run_root / "predictions"

    tasks = ("binary", "ternary")
    experiments = {
        task: task_experiment_name(args.track, subtrack_dir, task, run_id)
        for task in tasks
    }
    checkpoints = {
        task: PROJECT_ROOT / args.checkpoints_dir / args.track / subtrack_dir / task / experiments[task]
        for task in tasks
    }
    train_logs = {
        task: PROJECT_ROOT / args.train_logs_dir / args.track / subtrack_dir / task / experiments[task]
        for task in tasks
    }
    test_logs = {
        task: PROJECT_ROOT / args.test_logs_dir / args.track / subtrack_dir / task / experiments[task]
        for task in tasks
    }

    train_scripts = {
        task: PROJECT_ROOT / "scripts" / args.track / subtrack_dir / f"run_{task}.sh"
        for task in tasks
    }
    test_scripts = {
        task: PROJECT_ROOT / "test_scripts" / args.track / subtrack_dir / f"run_{task}.sh"
        for task in tasks
    }

    missing_scripts = [path for path in [*train_scripts.values(), *test_scripts.values()] if not path.exists()]
    if missing_scripts:
        missing = "\n".join(str(path.relative_to(PROJECT_ROOT)) for path in missing_scripts)
        raise FileNotFoundError(f"Missing pipeline script(s):\n{missing}")

    if args.dry_run:
        print(f"Run root: {rel_posix(run_root)}")
        for task in tasks:
            if not args.skip_train:
                print(
                    "TRAIN",
                    task,
                    f"EXPERIMENT_NAME={experiments[task]}",
                    f"LOGS_DIR={args.train_logs_dir}",
                    args.bash_bin,
                    rel_posix(train_scripts[task]),
                )
        if not args.skip_test:
            for task in tasks:
                print(
                    "TEST",
                    task,
                    f"CHECKPOINT_DIR={rel_posix(checkpoints[task])}",
                    f"LOGS_DIR={args.test_logs_dir}",
                    args.bash_bin,
                    rel_posix(test_scripts[task]),
                    "--prediction_csv",
                    rel_posix(predictions_dir / f"{task}.csv"),
                )
            print(
                "PACKAGE",
                args.python_bin,
                "make/make.py",
                "--binary_csv",
                rel_posix(predictions_dir / "binary.csv"),
                "--ternary_csv",
                rel_posix(predictions_dir / "ternary.csv"),
                "--output_dir",
                rel_posix(predictions_dir),
            )
        return

    normalize_shell_line_endings(
        [
            *train_scripts.values(),
            *test_scripts.values(),
            PROJECT_ROOT / "test_scripts" / "_common.sh",
        ]
    )

    predictions_dir.mkdir(parents=True, exist_ok=True)

    common_env = os.environ.copy()
    common_env.update(
        {
            "PYTHON_BIN": args.python_bin,
            "DEVICE": args.device,
            "CHECKPOINTS_DIR": args.checkpoints_dir,
        }
    )

    failures: list[str] = []
    for task in tasks:
        if args.skip_train:
            continue
        env = common_env.copy()
        env.update(
            {
                "EXPERIMENT_NAME": experiments[task],
                "LOGS_DIR": args.train_logs_dir,
            }
        )
        try:
            run_command([args.bash_bin, rel_posix(train_scripts[task])], env)
        except subprocess.CalledProcessError as exc:
            failures.append(f"train {task}: exit code {exc.returncode}")
            if not args.keep_going:
                raise

    if failures:
        raise RuntimeError("; ".join(failures))

    if not args.skip_test:
        for task in tasks:
            env = common_env.copy()
            env.update(
                {
                    "CHECKPOINT_DIR": rel_posix(checkpoints[task]),
                    "LOGS_DIR": args.test_logs_dir,
                }
            )
            prediction_csv = predictions_dir / f"{task}.csv"
            run_command(
                [args.bash_bin, rel_posix(test_scripts[task]), "--prediction_csv", rel_posix(prediction_csv)],
                env,
            )
            if not prediction_csv.exists():
                root_prediction_csv = PROJECT_ROOT / "predictions" / f"{task}.csv"
                if root_prediction_csv.exists():
                    raise FileNotFoundError(
                        f"Expected {rel_posix(prediction_csv)}, but test.py wrote {rel_posix(root_prediction_csv)}. "
                        "Please use the updated test.py path remapping."
                    )
                raise FileNotFoundError(f"Expected prediction CSV was not generated: {rel_posix(prediction_csv)}")

        make_script = PROJECT_ROOT / "make" / "make.py"
        run_command(
            [
                args.python_bin,
                rel_posix(make_script),
                "--binary_csv",
                rel_posix(predictions_dir / "binary.csv"),
                "--ternary_csv",
                rel_posix(predictions_dir / "ternary.csv"),
                "--output_dir",
                rel_posix(predictions_dir),
            ],
            common_env,
        )
        submission_zip = predictions_dir / "submission.zip"
        if not submission_zip.exists():
            raise FileNotFoundError(f"Expected submission zip was not generated: {submission_zip}")

    manifest = {
        "run_id": run_id,
        "track": args.track,
        "subtrack": subtrack_train_name,
        "subtrack_dir": subtrack_dir,
        "device": args.device,
        "experiments": experiments,
        "checkpoint_dirs": {task: rel_posix(path) for task, path in checkpoints.items()},
        "train_log_dirs": {task: rel_posix(path) for task, path in train_logs.items()},
        "test_log_dirs": {task: rel_posix(path) for task, path in test_logs.items()},
        "predictions_dir": rel_posix(predictions_dir),
        "submission_zip": rel_posix(predictions_dir / "submission.zip"),
    }
    manifest_path = run_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("\nPipeline finished.", flush=True)
    if not args.skip_test:
        print(f"Prediction folder zip: {predictions_dir / 'submission.zip'}", flush=True)
    if args.archive_artifacts:
        archive_path = run_root / f"{args.track}_{subtrack_dir}_{run_id}_artifacts.zip"
        write_artifact_archive(archive_path, manifest_path, run_root, checkpoints, train_logs, test_logs)
        print(f"Artifact archive: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
