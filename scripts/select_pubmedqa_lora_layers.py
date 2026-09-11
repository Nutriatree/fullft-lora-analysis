#!/usr/bin/env python3
"""Select high- and low-update LoRA layers from an L1 final checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pubmedqa.experiment_runs import DEFAULT_SHARED_DEFAULTS
from pubmedqa.data.records import safe_name, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default=DEFAULT_SHARED_DEFAULTS.model_name)
    parser.add_argument("--condition", default="lora")
    parser.add_argument("--train-output-dir", type=Path, default=Path("outputs/pubmedqa_train"))
    parser.add_argument("--layer-count", type=int, default=4)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_final_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [record for record in records if float(record.get("checkpoint_percent", 0.0)) >= 100.0]
    if not candidates:
        raise RuntimeError("L1 has no 100% checkpoint record.")
    return max(candidates, key=lambda record: int(record["global_step"]))


def _aggregate_layer_scores(records: list[dict[str, Any]]) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for record in records:
        layer_index = record.get("layer_index")
        if layer_index is None:
            continue
        scores[int(layer_index)] += float(record["relative_update_norm"])
    if not scores:
        raise RuntimeError("No layer-wise LoRA update records were found in the selected checkpoint.")
    return dict(scores)


def _selection_payload(
    *,
    scope: str,
    layers: list[int],
    layer_scores: dict[int, float],
    layer_count: int,
    source_checkpoint: dict[str, Any],
    source_file: Path,
) -> dict[str, Any]:
    return {
        "layer_scope": scope,
        "target_layers": layers,
        "selection_metric": "sum(relative_update_norm) across L1 target modules per layer",
        "layer_count": layer_count,
        "source_checkpoint": {
            "checkpoint_kind": source_checkpoint["checkpoint_kind"],
            "checkpoint_percent": source_checkpoint["checkpoint_percent"],
            "epoch": source_checkpoint["epoch"],
            "global_step": source_checkpoint["global_step"],
            "checkpoint_dir": source_checkpoint["checkpoint_dir"],
        },
        "source_layerwise_file": str(source_file),
        "layer_scores": {str(layer): layer_scores[layer] for layer in sorted(layer_scores)},
    }


def main() -> None:
    args = parse_args()
    if args.layer_count <= 0:
        raise ValueError("--layer-count must be positive.")

    l1_root = (
        args.train_output_dir
        / args.run_id
        / safe_name(args.model_name)
        / safe_name(args.condition)
    )
    checkpoint_log = l1_root / "logs" / "checkpoints.jsonl"
    if not checkpoint_log.is_file():
        raise FileNotFoundError(f"L1 checkpoint log not found: {checkpoint_log}")

    checkpoint = _select_final_checkpoint(_read_jsonl(checkpoint_log))
    layerwise_file = l1_root / "layerwise_updates" / f"{Path(checkpoint['checkpoint_dir']).name}.jsonl"
    if not layerwise_file.is_file():
        raise FileNotFoundError(f"L1 layer-wise artifact not found: {layerwise_file}")

    scores = _aggregate_layer_scores(_read_jsonl(layerwise_file))
    ranked_high = sorted(scores, key=lambda layer: (-scores[layer], layer))
    ranked_low = sorted(scores, key=lambda layer: (scores[layer], layer))
    if args.layer_count * 2 > len(scores):
        raise ValueError(
            f"--layer-count={args.layer_count} requires at least {args.layer_count * 2} tracked layers; "
            f"found {len(scores)}."
        )

    high_layers = sorted(ranked_high[: args.layer_count])
    low_layers = sorted(ranked_low[: args.layer_count])
    if set(high_layers) & set(low_layers):
        raise RuntimeError("High- and low-update selections overlap; decrease --layer-count.")

    analysis_dir = l1_root / "analysis"
    high_path = analysis_dir / "selective_layers_high.json"
    low_path = analysis_dir / "selective_layers_low.json"
    write_json(
        high_path,
        _selection_payload(
            scope="high-update",
            layers=high_layers,
            layer_scores=scores,
            layer_count=args.layer_count,
            source_checkpoint=checkpoint,
            source_file=layerwise_file,
        ),
    )
    write_json(
        low_path,
        _selection_payload(
            scope="low-update",
            layers=low_layers,
            layer_scores=scores,
            layer_count=args.layer_count,
            source_checkpoint=checkpoint,
            source_file=layerwise_file,
        ),
    )
    print(f"[high-update] layers={','.join(map(str, high_layers))} file={high_path}")
    print(f"[low-update] layers={','.join(map(str, low_layers))} file={low_path}")


if __name__ == "__main__":
    main()
