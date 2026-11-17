"""Non-invasive T2-A/T2-B diagnostics for ProtoV2.2.4.

The accumulator receives predictions and gates produced by the existing global
evaluation forward.  It never invokes a model, a loader, or an RNG function.
Only aggregate integer counts are retained; sample logits are never saved.
"""

import csv
import os
from pathlib import Path

import numpy as np


BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _empty_counts(size):
    return {
        key: np.zeros(size, dtype=np.int64)
        for key in (
            "samples",
            "linear_correct",
            "proto_correct",
            "fused_correct",
            "correction",
            "harm",
        )
    }


def _bin_indices(values):
    """Map [0, 1] values to the five pre-registered left-closed bins."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.searchsorted(BIN_EDGES[1:-1], values, side="right")


def _add_group_counts(storage, groups, masks, size):
    storage["samples"] += np.bincount(groups, minlength=size)
    for key, mask in masks.items():
        storage[key] += np.bincount(groups[mask], minlength=size)


class T2EvaluationAccumulator:
    """Accumulate exact aggregate diagnostics from one existing eval pass."""

    def __init__(self, num_classes):
        self.num_classes = int(num_classes)
        self.linear_correct = 0
        self.proto_correct = 0
        self.fused_correct = 0
        self.total = 0
        self.both_correct = 0
        self.linear_only_correct = 0
        self.proto_only_correct = 0
        self.both_wrong = 0
        self.correction = 0
        self.harm = 0
        self.entropy_bins = _empty_counts(5)
        self.recommendation_reliability_bins = _empty_counts(5)
        self.true_class_reliability_bins = _empty_counts(5)
        self.joint_bins = _empty_counts(25)
        self.class_correct_fused = np.zeros(
            self.num_classes, dtype=np.int64
        )
        self.class_total = np.zeros(self.num_classes, dtype=np.int64)

    def update(
        self,
        targets,
        linear_predictions,
        proto_predictions,
        fused_predictions,
        normalized_linear_entropy,
        class_reliability,
    ):
        targets = targets.detach().cpu().numpy().astype(np.int64, copy=False)
        linear_predictions = (
            linear_predictions.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        proto_predictions = (
            proto_predictions.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        fused_predictions = (
            fused_predictions.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        entropy = (
            normalized_linear_entropy.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
        )
        reliability = (
            class_reliability.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
        )

        linear_ok = linear_predictions == targets
        proto_ok = proto_predictions == targets
        fused_ok = fused_predictions == targets
        correction = np.logical_and(~linear_ok, fused_ok)
        harm = np.logical_and(linear_ok, ~fused_ok)
        masks = {
            "linear_correct": linear_ok,
            "proto_correct": proto_ok,
            "fused_correct": fused_ok,
            "correction": correction,
            "harm": harm,
        }

        self.linear_correct += int(linear_ok.sum())
        self.proto_correct += int(proto_ok.sum())
        self.fused_correct += int(fused_ok.sum())
        self.total += int(targets.size)
        self.both_correct += int(np.logical_and(linear_ok, proto_ok).sum())
        self.linear_only_correct += int(
            np.logical_and(linear_ok, ~proto_ok).sum()
        )
        self.proto_only_correct += int(
            np.logical_and(~linear_ok, proto_ok).sum()
        )
        self.both_wrong += int(np.logical_and(~linear_ok, ~proto_ok).sum())
        self.correction += int(correction.sum())
        self.harm += int(harm.sum())

        entropy_bin = _bin_indices(entropy)
        # Research-only per-sample attribution: this is the reliability of the
        # class recommended by the prototype branch.  It is NOT an original
        # scalar reliability gate used by the ProtoV2.2.4 algorithm.
        recommendation_reliability = reliability[proto_predictions]
        true_class_reliability = reliability[targets]
        recommendation_bin = _bin_indices(recommendation_reliability)
        true_class_bin = _bin_indices(true_class_reliability)
        joint_bin = entropy_bin * 5 + recommendation_bin

        _add_group_counts(self.entropy_bins, entropy_bin, masks, 5)
        _add_group_counts(
            self.recommendation_reliability_bins,
            recommendation_bin,
            masks,
            5,
        )
        _add_group_counts(
            self.true_class_reliability_bins,
            true_class_bin,
            masks,
            5,
        )
        _add_group_counts(self.joint_bins, joint_bin, masks, 25)

        self.class_total += np.bincount(
            targets, minlength=self.num_classes
        )
        self.class_correct_fused += np.bincount(
            targets[fused_ok], minlength=self.num_classes
        )

    def finalize(self):
        lhs = self.fused_correct - self.linear_correct
        rhs = self.correction - self.harm
        if lhs != rhs:
            raise AssertionError(
                "T2-A identity failed: fused_correct-linear_correct "
                f"({lhs}) != correction-harm ({rhs})"
            )
        return {
            "linear_correct": self.linear_correct,
            "proto_correct": self.proto_correct,
            "fused_correct": self.fused_correct,
            "total": self.total,
            "both_correct": self.both_correct,
            "linear_only_correct": self.linear_only_correct,
            "proto_only_correct": self.proto_only_correct,
            "both_wrong": self.both_wrong,
            "correction": self.correction,
            "harm": self.harm,
            "identity_lhs": lhs,
            "identity_rhs": rhs,
            "entropy_bins": self.entropy_bins,
            "recommendation_reliability_bins": (
                self.recommendation_reliability_bins
            ),
            "true_class_reliability_bins": (
                self.true_class_reliability_bins
            ),
            "joint_bins": self.joint_bins,
            "class_correct_fused": self.class_correct_fused,
            "class_total": self.class_total,
        }


class T2DiagnosticsWriter:
    """Durably append pre-registered aggregate CSV diagnostics."""

    FILE_FIELDS = {
        "t2a_round_summary.csv": [
            "round",
            "linear_correct",
            "proto_correct",
            "fused_correct",
            "total",
            "correction",
            "harm",
            "identity_lhs",
            "identity_rhs",
            "identity_valid",
        ],
        "t2a_complementarity.csv": [
            "round",
            "both_correct",
            "linear_only_correct",
            "proto_only_correct",
            "both_wrong",
            "correction",
            "harm",
            "total",
        ],
        "t2a_entropy_bins.csv": [
            "round",
            "bin_index",
            "bin_left",
            "bin_right",
            "right_closed",
            "samples",
            "linear_correct",
            "proto_correct",
            "fused_correct",
            "correction",
            "harm",
        ],
        "t2a_reliability_bins.csv": [
            "round",
            "reliability_kind",
            "bin_index",
            "bin_left",
            "bin_right",
            "right_closed",
            "samples",
            "linear_correct",
            "proto_correct",
            "fused_correct",
            "correction",
            "harm",
        ],
        "t2a_joint_bins.csv": [
            "round",
            "entropy_bin_index",
            "reliability_bin_index",
            "entropy_left",
            "entropy_right",
            "reliability_left",
            "reliability_right",
            "samples",
            "linear_correct",
            "proto_correct",
            "fused_correct",
            "correction",
            "harm",
        ],
        "t2b_round_class_support.csv": [
            "round",
            "class_id",
            "support_count",
        ],
        "t2b_class_outcomes.csv": [
            "round",
            "class_id",
            "class_correct_fused",
            "class_total",
        ],
    }

    def __init__(self, directory, num_classes, overwrite=False):
        self.directory = Path(directory)
        self.num_classes = int(num_classes)
        self.overwrite = bool(overwrite)

    def prepare(self, completed_round, resuming):
        self.directory.mkdir(parents=True, exist_ok=True)
        for filename, fields in self.FILE_FIELDS.items():
            path = self.directory / filename
            if not path.exists():
                continue
            if not resuming:
                if self.overwrite:
                    path.unlink()
                    continue
                raise FileExistsError(
                    f"diagnostic file already exists: {path}"
                )
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if int(row["round"]) <= int(completed_round)
                ]
            self._rewrite(path, fields, rows)

    def write_support(self, round_number, support_counts):
        if support_counts is None:
            rows = [
                {"round": 0, "class_id": class_id, "support_count": ""}
                for class_id in range(self.num_classes)
            ]
        else:
            counts = np.asarray(support_counts, dtype=np.int64)
            if counts.shape != (self.num_classes,):
                raise ValueError(
                    "round-class support has wrong shape: "
                    f"{counts.shape} != {(self.num_classes,)}"
                )
            rows = [
                {
                    "round": int(round_number),
                    "class_id": class_id,
                    "support_count": int(counts[class_id]),
                }
                for class_id in range(self.num_classes)
            ]
        self._append("t2b_round_class_support.csv", rows)

    def write_evaluation(self, round_number, result):
        round_number = int(round_number)
        self._append(
            "t2a_round_summary.csv",
            [
                {
                    "round": round_number,
                    "linear_correct": result["linear_correct"],
                    "proto_correct": result["proto_correct"],
                    "fused_correct": result["fused_correct"],
                    "total": result["total"],
                    "correction": result["correction"],
                    "harm": result["harm"],
                    "identity_lhs": result["identity_lhs"],
                    "identity_rhs": result["identity_rhs"],
                    "identity_valid": int(
                        result["identity_lhs"] == result["identity_rhs"]
                    ),
                }
            ],
        )
        self._append(
            "t2a_complementarity.csv",
            [
                {
                    "round": round_number,
                    "both_correct": result["both_correct"],
                    "linear_only_correct": result["linear_only_correct"],
                    "proto_only_correct": result["proto_only_correct"],
                    "both_wrong": result["both_wrong"],
                    "correction": result["correction"],
                    "harm": result["harm"],
                    "total": result["total"],
                }
            ],
        )
        self._write_bin_rows(
            "t2a_entropy_bins.csv",
            round_number,
            result["entropy_bins"],
        )
        self._write_bin_rows(
            "t2a_reliability_bins.csv",
            round_number,
            result["recommendation_reliability_bins"],
            reliability_kind="proto_recommendation_reliability",
        )
        self._write_bin_rows(
            "t2a_reliability_bins.csv",
            round_number,
            result["true_class_reliability_bins"],
            reliability_kind="true_class_reliability",
        )
        self._write_joint_rows(round_number, result["joint_bins"])
        self._append(
            "t2b_class_outcomes.csv",
            [
                {
                    "round": round_number,
                    "class_id": class_id,
                    "class_correct_fused": int(
                        result["class_correct_fused"][class_id]
                    ),
                    "class_total": int(result["class_total"][class_id]),
                }
                for class_id in range(self.num_classes)
            ],
        )

    def _write_bin_rows(
        self,
        filename,
        round_number,
        counts,
        reliability_kind=None,
    ):
        rows = []
        for index in range(5):
            row = {
                "round": round_number,
                "bin_index": index,
                "bin_left": BIN_EDGES[index],
                "bin_right": BIN_EDGES[index + 1],
                "right_closed": int(index == 4),
            }
            if reliability_kind is not None:
                row["reliability_kind"] = reliability_kind
            for key in counts:
                row[key] = int(counts[key][index])
            rows.append(row)
        self._append(filename, rows)

    def _write_joint_rows(self, round_number, counts):
        rows = []
        for entropy_index in range(5):
            for reliability_index in range(5):
                flat_index = entropy_index * 5 + reliability_index
                row = {
                    "round": round_number,
                    "entropy_bin_index": entropy_index,
                    "reliability_bin_index": reliability_index,
                    "entropy_left": BIN_EDGES[entropy_index],
                    "entropy_right": BIN_EDGES[entropy_index + 1],
                    "reliability_left": BIN_EDGES[reliability_index],
                    "reliability_right": BIN_EDGES[reliability_index + 1],
                }
                for key in counts:
                    row[key] = int(counts[key][flat_index])
                rows.append(row)
        self._append("t2a_joint_bins.csv", rows)

    def _append(self, filename, rows):
        if not rows:
            return
        path = self.directory / filename
        fields = self.FILE_FIELDS[filename]
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
