#!/usr/bin/env python3
"""Validate and combine scalar Earthquake Malliavin teacher shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import torch


def _load_shard(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing teacher shard: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"teacher shard must be a dictionary: {path}")
    required = {
        "format_version",
        "teacher",
        "start",
        "end",
        "total_size",
        "train_size",
        "validation_size",
        "dataset_keys",
        "detail_keys",
        "dtype",
        "global_indices",
        "dataset",
        "teacher_details",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"teacher shard {path} is missing fields: {missing}")
    payload = dict(payload)
    payload["_path"] = str(path)
    return payload


def _validate_tensor_group(
    shard: Dict[str, object],
    *,
    group_name: str,
    key_field: str,
    expected_dtype: torch.dtype,
) -> None:
    path = shard["_path"]
    start = int(shard["start"])
    end = int(shard["end"])
    shard_size = end - start
    keys = list(shard[key_field])
    group = shard[group_name]
    if not isinstance(group, dict):
        raise ValueError(f"{group_name} must be a dictionary in {path}")
    if list(group) != keys:
        raise ValueError(
            f"{group_name} keys/order do not match {key_field} in {path}"
        )
    for key in keys:
        tensor = group[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{group_name}.{key} is not a tensor in {path}")
        if tensor.ndim < 1 or tensor.shape[0] != shard_size:
            raise ValueError(
                f"{group_name}.{key} first dimension must be {shard_size} "
                f"in {path}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"{group_name}.{key} dtype must be {expected_dtype} "
                f"in {path}, got {tensor.dtype}"
            )


def combine_teacher_shards(
    shard_paths: Sequence[Path],
    *,
    output_dir: Path,
) -> Dict[str, object]:
    """Combine complete, non-overlapping shards in original global-index order."""

    if not shard_paths:
        raise ValueError("at least one teacher shard is required")
    shards = sorted(
        (_load_shard(Path(path)) for path in shard_paths),
        key=lambda payload: int(payload["start"]),
    )
    reference = shards[0]
    metadata_fields = (
        "format_version",
        "teacher",
        "total_size",
        "train_size",
        "validation_size",
        "dataset_keys",
        "detail_keys",
        "dtype",
    )
    for shard in shards[1:]:
        for field in metadata_fields:
            if shard[field] != reference[field]:
                raise ValueError(
                    f"shard metadata mismatch for {field}: "
                    f"{reference['_path']} vs {shard['_path']}"
                )

    if reference["format_version"] != 1:
        raise ValueError(f"unsupported shard format: {reference['format_version']}")
    if reference["teacher"] != "malliavin":
        raise ValueError("only Malliavin teacher shards are supported")
    dtype_name = str(reference["dtype"])
    try:
        expected_dtype = getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"unsupported shard dtype: {dtype_name!r}") from exc
    if not isinstance(expected_dtype, torch.dtype):
        raise ValueError(f"unsupported shard dtype: {dtype_name!r}")

    total_size = int(reference["total_size"])
    train_size = int(reference["train_size"])
    validation_size = int(reference["validation_size"])
    if train_size + validation_size != total_size:
        raise ValueError("train_size + validation_size does not equal total_size")

    expected_start = 0
    all_indices = []
    for shard in shards:
        start = int(shard["start"])
        end = int(shard["end"])
        if start < expected_start:
            raise ValueError(
                f"overlapping teacher shard at [{start}, {end}); "
                f"next expected index is {expected_start}"
            )
        if start > expected_start:
            raise ValueError(
                f"missing teacher indices [{expected_start}, {start})"
            )
        if end <= start or end > total_size:
            raise ValueError(f"invalid teacher shard range [{start}, {end})")
        indices = shard["global_indices"]
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64:
            raise ValueError(f"global_indices must be an int64 tensor in {shard['_path']}")
        expected_indices = torch.arange(start, end, dtype=torch.int64)
        if not torch.equal(indices, expected_indices):
            raise ValueError(
                f"global_indices are missing, duplicated, or out of order in "
                f"{shard['_path']}"
            )
        _validate_tensor_group(
            shard,
            group_name="dataset",
            key_field="dataset_keys",
            expected_dtype=expected_dtype,
        )
        _validate_tensor_group(
            shard,
            group_name="teacher_details",
            key_field="detail_keys",
            expected_dtype=expected_dtype,
        )
        all_indices.append(indices)
        expected_start = end
    if expected_start != total_size:
        raise ValueError(f"missing teacher indices [{expected_start}, {total_size})")
    combined_indices = torch.cat(all_indices)
    if not torch.equal(combined_indices, torch.arange(total_size, dtype=torch.int64)):
        raise ValueError("combined teacher indices are not in original order")

    for group_name, key_field in (
        ("dataset", "dataset_keys"),
        ("teacher_details", "detail_keys"),
    ):
        for key in reference[key_field]:
            tail_shape = tuple(reference[group_name][key].shape[1:])
            for shard in shards[1:]:
                actual_tail = tuple(shard[group_name][key].shape[1:])
                if actual_tail != tail_shape:
                    raise ValueError(
                        f"shape mismatch for {group_name}.{key}: "
                        f"expected (*, {tail_shape}), got (*, {actual_tail}) "
                        f"in {shard['_path']}"
                    )

    combined_dataset = {
        key: torch.cat([shard["dataset"][key] for shard in shards], dim=0)
        for key in reference["dataset_keys"]
    }
    combined_details = {
        key: torch.cat(
            [shard["teacher_details"][key] for shard in shards], dim=0
        )
        for key in reference["detail_keys"]
    }
    train_dataset = {
        key: value[:train_size] for key, value in combined_dataset.items()
    }
    validation_dataset = {
        key: value[train_size:] for key, value in combined_dataset.items()
    }
    train_details = {
        key: value[:train_size] for key, value in combined_details.items()
    }
    validation_details = {
        key: value[train_size:] for key, value in combined_details.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_dataset, output_dir / "teacher_dataset.pt")
    torch.save(validation_dataset, output_dir / "validation_dataset.pt")
    torch.save(train_details, output_dir / "teacher_details.pt")
    torch.save(validation_details, output_dir / "validation_teacher_details.pt")
    manifest = {
        "format_version": 1,
        "teacher": "malliavin",
        "dtype": dtype_name,
        "total_size": total_size,
        "train_size": train_size,
        "validation_size": validation_size,
        "dataset_keys": list(reference["dataset_keys"]),
        "detail_keys": list(reference["detail_keys"]),
        "shards": [
            {
                "path": shard["_path"],
                "start": int(shard["start"]),
                "end": int(shard["end"]),
            }
            for shard in shards
        ],
        "outputs": {
            "teacher_dataset": str(output_dir / "teacher_dataset.pt"),
            "validation_dataset": str(output_dir / "validation_dataset.pt"),
            "teacher_details": str(output_dir / "teacher_details.pt"),
            "validation_teacher_details": str(
                output_dir / "validation_teacher_details.pt"
            ),
        },
    }
    with (output_dir / "teacher_shard_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    return {
        "manifest": manifest,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "train_details": train_details,
        "validation_details": validation_details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = combine_teacher_shards(args.shards, output_dir=args.output_dir.resolve())
    print(json.dumps(result["manifest"], indent=2))


if __name__ == "__main__":
    main()
