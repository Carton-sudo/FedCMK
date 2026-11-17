"""Single-pass per-sample prediction logging for paired Proto ablations.

The accumulator consumes predictions already produced by the existing global
server evaluation.  It does not traverse a loader, invoke a model, or call an
RNG function.  Only integer labels/predictions and deterministic integrity
hashes are persisted; logits and feature tensors are intentionally excluded.
"""

import csv
import hashlib
import os
from pathlib import Path

import numpy as np


PREDICTION_FIELDS = [
    "round",
    "sample_id",
    "target",
    "linear_pred",
    "proto_pred",
    "fused_pred",
    "main_pred",
]

MANIFEST_FIELDS = [
    "round",
    "proto_arm",
    "sample_count",
    "sample_id_min",
    "sample_id_max",
    "target_order_sha256",
    "prediction_payload_sha256",
    "linear_correct",
    "proto_correct",
    "fused_correct",
    "main_correct",
]


def _as_int64(values):
    return values.detach().cpu().numpy().astype(np.int64, copy=False)


def _little_endian_bytes(values):
    return np.asarray(values, dtype="<i8").tobytes(order="C")


class SamplePredictionAccumulator:
    """Collect exact sample identities and predictions from one eval pass."""

    def __init__(self):
        self._next_sample_id = 0
        self._parts = {
            key: []
            for key in (
                "sample_id",
                "target",
                "linear_pred",
                "proto_pred",
                "fused_pred",
                "main_pred",
            )
        }

    def update(
        self,
        targets,
        linear_predictions,
        proto_predictions,
        fused_predictions,
        main_predictions,
    ):
        arrays = {
            "target": _as_int64(targets),
            "linear_pred": _as_int64(linear_predictions),
            "proto_pred": _as_int64(proto_predictions),
            "fused_pred": _as_int64(fused_predictions),
            "main_pred": _as_int64(main_predictions),
        }
        sizes = {array.size for array in arrays.values()}
        if len(sizes) != 1:
            raise ValueError("per-sample prediction batch sizes do not match")
        size = sizes.pop()
        sample_ids = np.arange(
            self._next_sample_id,
            self._next_sample_id + size,
            dtype=np.int64,
        )
        self._next_sample_id += size
        self._parts["sample_id"].append(sample_ids)
        for key, array in arrays.items():
            self._parts[key].append(array)

    def finalize(self):
        result = {
            key: (
                np.concatenate(parts)
                if parts
                else np.empty(0, dtype=np.int64)
            )
            for key, parts in self._parts.items()
        }
        sample_count = int(result["target"].size)
        expected_ids = np.arange(sample_count, dtype=np.int64)
        if not np.array_equal(result["sample_id"], expected_ids):
            raise AssertionError(
                "sample IDs are not the contiguous global-test traversal order"
            )
        result["sample_count"] = sample_count
        result["target_order_sha256"] = hashlib.sha256(
            _little_endian_bytes(result["sample_id"])
            + _little_endian_bytes(result["target"])
        ).hexdigest()
        payload = np.column_stack(
            [
                result["sample_id"],
                result["target"],
                result["linear_pred"],
                result["proto_pred"],
                result["fused_pred"],
                result["main_pred"],
            ]
        )
        result["prediction_payload_sha256"] = hashlib.sha256(
            _little_endian_bytes(payload)
        ).hexdigest()
        return result


class SamplePredictionWriter:
    """Append exact per-sample predictions without touching prior run data."""

    PREDICTION_FILENAME = "paired_sample_predictions.csv"
    MANIFEST_FILENAME = "paired_sample_prediction_manifest.csv"

    def __init__(self, directory, start_round, overwrite=False):
        self.directory = Path(directory)
        self.start_round = int(start_round)
        self.overwrite = bool(overwrite)
        if self.start_round < 0:
            raise ValueError("sample_prediction_start_round must be >= 0")
        self._written_rounds = set()

    @property
    def prediction_path(self):
        return self.directory / self.PREDICTION_FILENAME

    @property
    def manifest_path(self):
        return self.directory / self.MANIFEST_FILENAME

    def prepare(self, completed_round, resuming):
        self.directory.mkdir(parents=True, exist_ok=True)
        completed_round = int(completed_round)
        for path, fields in (
            (self.prediction_path, PREDICTION_FIELDS),
            (self.manifest_path, MANIFEST_FIELDS),
        ):
            if not path.exists():
                continue
            if not resuming:
                if self.overwrite:
                    path.unlink()
                    continue
                raise FileExistsError(
                    f"sample prediction diagnostic already exists: {path}"
                )
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if int(row["round"]) <= completed_round
                ]
            self._rewrite(path, fields, rows)

        if self.manifest_path.exists():
            with self.manifest_path.open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                self._written_rounds = {
                    int(row["round"]) for row in csv.DictReader(handle)
                }

    def write_evaluation(self, round_number, proto_arm, result):
        round_number = int(round_number)
        if round_number < self.start_round:
            return False
        if round_number in self._written_rounds:
            raise ValueError(
                f"sample predictions already written for round {round_number}"
            )

        sample_count = int(result["sample_count"])
        rows = (
            {
                "round": round_number,
                "sample_id": int(result["sample_id"][index]),
                "target": int(result["target"][index]),
                "linear_pred": int(result["linear_pred"][index]),
                "proto_pred": int(result["proto_pred"][index]),
                "fused_pred": int(result["fused_pred"][index]),
                "main_pred": int(result["main_pred"][index]),
            }
            for index in range(sample_count)
        )
        self._append(self.prediction_path, PREDICTION_FIELDS, rows)

        target = result["target"]
        manifest = {
            "round": round_number,
            "proto_arm": str(proto_arm),
            "sample_count": sample_count,
            "sample_id_min": 0 if sample_count else "",
            "sample_id_max": sample_count - 1 if sample_count else "",
            "target_order_sha256": result["target_order_sha256"],
            "prediction_payload_sha256": result[
                "prediction_payload_sha256"
            ],
            "linear_correct": int((result["linear_pred"] == target).sum()),
            "proto_correct": int((result["proto_pred"] == target).sum()),
            "fused_correct": int((result["fused_pred"] == target).sum()),
            "main_correct": int((result["main_pred"] == target).sum()),
        }
        self._append(
            self.manifest_path,
            MANIFEST_FIELDS,
            [manifest],
        )
        self._written_rounds.add(round_number)
        return True

    @staticmethod
    def _append(path, fields, rows):
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _rewrite(path, fields, rows):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
