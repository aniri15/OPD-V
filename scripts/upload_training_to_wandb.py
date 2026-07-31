#!/usr/bin/env python3
"""Upload training artifacts to Weights & Biases.

Reads:
  - TensorBoard event files (verl training scalars)
  - Rollout JSONL dumps (compute MCQ accuracy + sample tables)

Usage:
  export WANDB_MODE=offline
  export WANDB_DIR=<WANDB_LOG_DIR>
  python scripts/upload_training_to_wandb.py \\
    --tensorboard-dir <TENSORBOARD_LOG_DIR>/OPD-V/OPD-V-Qwen3.5-4B \\
    --rollout-dir <ROLLOUT_DIR>/OPD-V-Qwen3.5-4B

Then sync:
  wandb sync <WANDB_LOG_DIR>/wandb/offline-run-*
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def extract_mcq_answer(text: str) -> str | None:
    """Extract A/B/C/D from model output (XML first, then heuristics)."""
    if not text:
        return None

    if "<answer>" in text:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        if answer in {"A", "B", "C", "D"}:
            return answer

    patterns = [
        r"(?:correct\s+)?answer\s*[:：]\s*\*?\*?\s*([A-D])\b",
        r"(?:option|choice)\s*[:：]?\s*([A-D])\b",
        r"\b([A-D])\.\s",
        r"(?:^|\n)\s*([A-D])\s*(?:\.|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()

    tail = text.strip().split("\n")[-1]
    m = re.search(r"\b([A-D])\b", tail)
    return m.group(1).upper() if m else None


def is_xml_format(text: str) -> bool:
    return re.search(r"<answer>\s*(A|B|C|D)\s*</answer>\s*$", text or "") is not None


def load_tensorboard_scalars(tensorboard_dir: Path) -> dict[str, list[tuple[int, float]]]:
    event_files = sorted(
        tensorboard_dir.glob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_size,
    )
    if not event_files:
        raise FileNotFoundError(f"No tensorboard event files under {tensorboard_dir}")

    ea = EventAccumulator(str(event_files[-1]))
    ea.Reload()
    scalars: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        scalars[tag] = [(e.step, float(e.value)) for e in ea.Scalars(tag)]
    return scalars


def load_wandb_history(source_wandb_run: Path) -> dict[int, dict]:
    """Read history rows from an offline run-*.wandb file."""
    history: dict[int, dict] = defaultdict(dict)
    datastore = DataStore()
    datastore.open_for_scan(str(source_wandb_run))
    try:
        while True:
            try:
                data = datastore.scan_data()
            except (AssertionError, EOFError):
                # Running jobs can leave a partial final record.
                break
            if data is None:
                break

            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue

            row = {}
            for item in record.history.item:
                key = "/".join(item.nested_key) if item.nested_key else item.key
                if not key:
                    continue
                try:
                    value = json.loads(item.value_json)
                except json.JSONDecodeError:
                    value = item.value_json
                row[key] = value

            step = row.get("training/global_step", row.get("_step", record.history.step.num))
            if step is None:
                continue
            history[int(step)].update(row)
    finally:
        datastore.close()

    return history


def parse_metric_value(value: str):
    value = value.strip()
    if value in {"True", "False"}:
        return value == "True"
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def load_console_history(source_log: Path) -> dict[int, dict]:
    """Parse `step:N - metric:value` console metric lines from slurm.out."""
    if not source_log or not source_log.exists():
        return {}

    history: dict[int, dict] = defaultdict(dict)
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    step_pattern = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+(.+)")

    with source_log.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = ansi_escape.sub("", line).strip()
            match = step_pattern.search(line)
            if not match:
                continue

            step = int(match.group(1))
            row = {"training/global_step": step}
            for part in match.group(2).split(" - "):
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                key = key.strip()
                if key:
                    row[key] = parse_metric_value(value)
            history[step].update(row)

    return history


def find_source_wandb_run(job_root: Path) -> Path | None:
    candidates = sorted((job_root / "wandb_logs" / "wandb").glob("offline-run-*/run-*.wandb"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_source_log(job_root: Path) -> Path | None:
    path = job_root / "logs" / "slurm.out"
    return path if path.exists() else None


def find_single_child_dir(parent: Path, label: str) -> Path:
    children = [p for p in parent.iterdir() if p.is_dir()]
    if not children:
        raise FileNotFoundError(f"No {label} directory under {parent}")
    if len(children) > 1:
        names = ", ".join(p.name for p in children)
        raise RuntimeError(f"Multiple {label} directories under {parent}: {names}")
    return children[0]


def compute_rollout_metrics(rollout_dir: Path) -> tuple[dict[int, dict[str, float]], dict[int, list[dict]]]:
    step_metrics: dict[int, dict[str, float]] = {}
    step_samples: dict[int, list[dict]] = defaultdict(list)

    jsonl_files = sorted(rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    for path in jsonl_files:
        step = int(path.stem)
        total = correct = format_ok = parsed = 0
        per_question: dict[str, list[bool]] = defaultdict(list)

        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Running jobs can leave a partially written final line.
                    continue
                gt = (row.get("gts") or "").strip().upper()
                output = row.get("output") or ""
                pred = extract_mcq_answer(output)
                is_correct = int(pred is not None and gt in {"A", "B", "C", "D"} and pred == gt)

                total += 1
                correct += is_correct
                format_ok += int(is_xml_format(output))
                parsed += int(pred is not None)

                key = row.get("input", "")[:200]
                per_question[key].append(bool(is_correct))

                if len(step_samples[step]) < 12:
                    step_samples[step].append(
                        {
                            "input": (row.get("input") or "")[:500],
                            "output": (row.get("output") or "")[:500],
                            "gts": gt,
                            "pred": pred,
                            "correct": is_correct,
                        }
                    )

        pass_at_1 = correct / total if total else 0.0
        pass_at_k = (
            sum(1 for vals in per_question.values() if any(vals)) / len(per_question)
            if per_question
            else 0.0
        )
        step_metrics[step] = {
            "rollout/acc_mean": pass_at_1,
            "rollout/acc_per_question_any": pass_at_k,
            "rollout/pred_parsed_rate": parsed / total if total else 0.0,
            "rollout/xml_format_rate": format_ok / total if total else 0.0,
            "rollout/num_samples": float(total),
        }

    return step_metrics, step_samples


def pick_scalar_groups(all_tags: list[str]) -> dict[str, list[str]]:
    groups = {
        "loss": [],
        "distillation": [],
        "rollout_corr": [],
        "performance": [],
        "response": [],
        "timing": [],
        "other": [],
    }
    for tag in all_tags:
        if tag.startswith("actor/") or tag.endswith("loss"):
            groups["loss"].append(tag)
        elif tag.startswith("self_distillation/"):
            groups["distillation"].append(tag)
        elif tag.startswith("rollout_corr/"):
            groups["rollout_corr"].append(tag)
        elif tag.startswith("perf/"):
            groups["performance"].append(tag)
        elif tag.startswith("response"):
            groups["response"].append(tag)
        elif tag.startswith("timing"):
            groups["timing"].append(tag)
        else:
            groups["other"].append(tag)
    return groups


def build_generation_table(step: int, samples: list[dict]) -> wandb.Table:
    table = wandb.Table(columns=["step", "input", "output", "gts", "pred", "correct"])
    for sample in samples:
        table.add_data(
            step,
            sample["input"],
            sample["output"],
            sample["gts"],
            sample["pred"],
            sample["correct"],
        )
    return table


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_cache = project_root.parent / "cache" / "opd-v"
    parser = argparse.ArgumentParser(description="Upload training logs to W&B")
    parser.add_argument(
        "--job-root",
        type=Path,
        help="Job directory containing rollouts/, wandb_logs/, and optionally tensorboard_log/.",
    )
    parser.add_argument(
        "--source-wandb-run",
        type=Path,
        help="Original offline run-*.wandb file to copy training metrics from.",
    )
    parser.add_argument(
        "--source-log",
        type=Path,
        help="slurm.out file to recover console-printed metrics missing from W&B history.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=default_cache / "tensorboard_log" / "OPD-V" / "OPD-V-Qwen3.5-4B",
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=default_cache / "rollouts" / "OPD-V-Qwen3.5-4B",
    )
    parser.add_argument("--project", default="opd-v")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--group", default="OPD-V-Qwen3.5-4B")
    parser.add_argument("--job-id", default=None)
    parser.add_argument(
        "--table-steps",
        default="1,10,50,100,130",
        help="Comma-separated rollout steps to log as wandb tables",
    )
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--allow-missing-tensorboard",
        action="store_true",
        help="Upload rollout metrics even if no TensorBoard event file exists.",
    )
    args = parser.parse_args()

    if args.job_root:
        args.job_root = args.job_root.resolve()
        if args.job_id is None:
            args.job_id = args.job_root.name.removeprefix("job_")
        if args.rollout_dir == parser.get_default("rollout_dir"):
            args.rollout_dir = find_single_child_dir(args.job_root / "rollouts", "rollout")
        if args.source_wandb_run is None:
            args.source_wandb_run = find_source_wandb_run(args.job_root)
        if args.source_log is None:
            args.source_log = find_source_log(args.job_root)
        if args.tensorboard_dir == parser.get_default("tensorboard_dir"):
            tensorboard_root = args.job_root / "tensorboard_log"
            if tensorboard_root.exists():
                args.tensorboard_dir = find_single_child_dir(tensorboard_root, "tensorboard project")
            else:
                args.tensorboard_dir = tensorboard_root

    if args.job_id is None:
        args.job_id = "unknown"
    if args.experiment_name is None:
        args.experiment_name = f"OPD-V-Qwen3.5-4B-job_{args.job_id}-posthoc"

    os.environ["WANDB_MODE"] = args.wandb_mode
    if "WANDB_DIR" not in os.environ:
        if args.job_root:
            os.environ["WANDB_DIR"] = str(args.job_root / "wandb_logs_posthoc")
        else:
            os.environ["WANDB_DIR"] = str(args.tensorboard_dir.parent.parent / "wandb_logs")

    source_history = load_wandb_history(args.source_wandb_run) if args.source_wandb_run else {}
    console_history = load_console_history(args.source_log) if args.source_log else {}

    if source_history:
        scalars = {}
    elif args.allow_missing_tensorboard and not args.tensorboard_dir.exists():
        scalars = {}
    else:
        try:
            scalars = load_tensorboard_scalars(args.tensorboard_dir)
        except FileNotFoundError:
            if not args.allow_missing_tensorboard:
                raise
            scalars = {}
    rollout_metrics, step_samples = compute_rollout_metrics(args.rollout_dir)
    table_steps = [int(s.strip()) for s in args.table_steps.split(",") if s.strip()]

    config = {
        "job_id": args.job_id,
        "tensorboard_dir": str(args.tensorboard_dir),
        "rollout_dir": str(args.rollout_dir),
        "source_wandb_run": str(args.source_wandb_run) if args.source_wandb_run else None,
        "source_log": str(args.source_log) if args.source_log else None,
        "num_source_wandb_steps": len(source_history),
        "num_console_steps": len(console_history),
        "num_scalar_tags": len(scalars),
        "num_rollout_steps": len(rollout_metrics),
    }

    wandb.init(
        project=args.project,
        name=args.experiment_name,
        group=args.group,
        config=config,
        tags=["opd-v", "post-hoc-upload", f"job-{args.job_id}"],
    )

    step_payload: dict[int, dict] = defaultdict(dict)
    for step, row in source_history.items():
        step_payload[step].update(row)
    for step, row in console_history.items():
        step_payload[step].update(row)
    for tag, points in scalars.items():
        for step, value in points:
            step_payload[step][tag] = value
    for step, metrics in rollout_metrics.items():
        step_payload[step].update(metrics)
    for step in table_steps:
        samples = step_samples.get(step)
        if samples:
            step_payload[step][f"rollout/generations_step_{step}"] = build_generation_table(step, samples)

    for step in sorted(step_payload):
        wandb.log(step_payload[step], step=step)

    # Summary panel values
    if rollout_metrics:
        last_step = max(rollout_metrics)
        wandb.summary["final_rollout_acc_mean"] = rollout_metrics[last_step]["rollout/acc_mean"]
        wandb.summary["final_rollout_acc_per_question_any"] = rollout_metrics[last_step][
            "rollout/acc_per_question_any"
        ]
    if "actor/vopd_loss" in scalars and scalars["actor/vopd_loss"]:
        wandb.summary["final_vopd_loss"] = scalars["actor/vopd_loss"][-1][1]
        wandb.summary["initial_vopd_loss"] = scalars["actor/vopd_loss"][0][1]

    print(f"Uploaded source W&B history for {len(source_history)} steps")
    print(f"Uploaded console history for {len(console_history)} steps")
    print(f"Uploaded {len(scalars)} tensorboard scalar series")
    print(f"Uploaded rollout metrics for {len(rollout_metrics)} steps")
    print(f"W&B run: {wandb.run.url if wandb.run else 'offline'}")

    wandb.finish()


if __name__ == "__main__":
    main()
