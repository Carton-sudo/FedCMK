"""Single-pass aggregate diagnostics for the Proto mechanism campaign.

The accumulator only consumes tensors already produced by the existing global
test forward.  It never invokes a model, a loader, or a random-number API.
No sample-level logits or features are persisted.
"""

import csv
import math
import os
from pathlib import Path

import numpy as np


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def _finite_or_blank(value):
    value = float(value)
    return value if math.isfinite(value) else ""


class ProtoGeometryAccumulator:
    """Aggregate geometry from the same batches used for server accuracy."""

    def __init__(self, num_classes, feature_dim):
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.count = np.zeros(self.num_classes, dtype=np.int64)
        self.feature_sum = np.zeros(
            (self.num_classes, self.feature_dim), dtype=np.float64
        )
        self.true_cosine_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.nearest_wrong_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.margin_sum = np.zeros(self.num_classes, dtype=np.float64)

    def update(self, targets, proto_features, proto_cosine_logits):
        targets = targets.detach().cpu().numpy().astype(np.int64, copy=False)
        features = proto_features.detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        cosine = proto_cosine_logits.detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature dimension changed: {features.shape[1]} != "
                f"{self.feature_dim}"
            )
        true_cosine = cosine[np.arange(targets.size), targets]
        wrong_cosine = cosine.copy()
        wrong_cosine[np.arange(targets.size), targets] = -np.inf
        nearest_wrong = wrong_cosine.max(axis=1)
        margin = true_cosine - nearest_wrong

        self.count += np.bincount(targets, minlength=self.num_classes)
        for class_id in range(self.num_classes):
            mask = targets == class_id
            if not mask.any():
                continue
            self.feature_sum[class_id] += features[mask].sum(axis=0)
            self.true_cosine_sum[class_id] += true_cosine[mask].sum()
            self.nearest_wrong_sum[class_id] += nearest_wrong[mask].sum()
            self.margin_sum[class_id] += margin[mask].sum()

    def finalize(self, prototypes, classifier_weight):
        prototypes = prototypes.detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        prototypes /= np.maximum(
            np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
        )
        valid = self.count > 0
        centroids = np.full_like(self.feature_sum, np.nan)
        centroid_norm = np.linalg.norm(self.feature_sum[valid], axis=1)
        centroids[valid] = self.feature_sum[valid] / np.maximum(
            centroid_norm[:, None], 1e-12
        )

        true_mean = np.full(self.num_classes, np.nan)
        wrong_mean = np.full(self.num_classes, np.nan)
        margin_mean = np.full(self.num_classes, np.nan)
        dispersion = np.full(self.num_classes, np.nan)
        true_mean[valid] = self.true_cosine_sum[valid] / self.count[valid]
        wrong_mean[valid] = self.nearest_wrong_sum[valid] / self.count[valid]
        margin_mean[valid] = self.margin_sum[valid] / self.count[valid]
        dispersion[valid] = np.maximum(
            2.0
            - 2.0
            * np.linalg.norm(self.feature_sum[valid], axis=1)
            / self.count[valid],
            0.0,
        )

        centroid_nearest = np.full(self.num_classes, np.nan)
        centroid_mean_other = np.full(self.num_classes, np.nan)
        valid_ids = np.flatnonzero(valid)
        centroid_pair_values = np.asarray([], dtype=np.float64)
        if valid_ids.size >= 2:
            matrix = centroids[valid_ids] @ centroids[valid_ids].T
            off_diagonal = ~np.eye(valid_ids.size, dtype=bool)
            centroid_pair_values = matrix[off_diagonal]
            for row_index, class_id in enumerate(valid_ids):
                others = np.delete(matrix[row_index], row_index)
                centroid_nearest[class_id] = float(others.max())
                centroid_mean_other[class_id] = float(others.mean())

        proto_matrix = prototypes @ prototypes.T
        proto_off_diagonal = proto_matrix[
            ~np.eye(self.num_classes, dtype=bool)
        ]

        weight_proto = np.full(self.num_classes, np.nan)
        weight_compatible = (
            classifier_weight is not None
            and tuple(classifier_weight.shape) == tuple(prototypes.shape)
        )
        if weight_compatible:
            weights = classifier_weight.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            weights /= np.maximum(
                np.linalg.norm(weights, axis=1, keepdims=True), 1e-12
            )
            weight_proto = np.sum(weights * prototypes, axis=1)

        total = int(self.count.sum())
        summary = {
            "total": total,
            "true_class_cosine_micro": _safe_divide(
                self.true_cosine_sum.sum(), total
            ),
            "nearest_wrong_cosine_micro": _safe_divide(
                self.nearest_wrong_sum.sum(), total
            ),
            "proto_margin_micro": _safe_divide(self.margin_sum.sum(), total),
            "true_class_cosine_macro": float(np.nanmean(true_mean)),
            "nearest_wrong_cosine_macro": float(np.nanmean(wrong_mean)),
            "proto_margin_macro": float(np.nanmean(margin_mean)),
            "within_class_dispersion_macro": float(np.nanmean(dispersion)),
            "centroid_pair_cosine_mean": (
                float(centroid_pair_values.mean())
                if centroid_pair_values.size
                else float("nan")
            ),
            "centroid_pair_cosine_max": (
                float(centroid_pair_values.max())
                if centroid_pair_values.size
                else float("nan")
            ),
            "inter_proto_cosine_mean": float(proto_off_diagonal.mean()),
            "inter_proto_cosine_min": float(proto_off_diagonal.min()),
            "inter_proto_cosine_max": float(proto_off_diagonal.max()),
            "classifier_weight_proto_cosine_mean": (
                float(np.nanmean(weight_proto))
                if weight_compatible
                else float("nan")
            ),
            "classifier_weight_proto_compatible": int(weight_compatible),
        }
        classwise = []
        for class_id in range(self.num_classes):
            classwise.append(
                {
                    "class_id": class_id,
                    "samples": int(self.count[class_id]),
                    "true_class_cosine": true_mean[class_id],
                    "nearest_wrong_cosine": wrong_mean[class_id],
                    "proto_margin": margin_mean[class_id],
                    "within_class_feature_dispersion": dispersion[class_id],
                    "centroid_nearest_wrong_cosine": centroid_nearest[class_id],
                    "centroid_mean_other_cosine": centroid_mean_other[class_id],
                    "classifier_weight_proto_cosine": weight_proto[class_id],
                }
            )
        return {"summary": summary, "classwise": classwise}


class ProtoCampaignDiagnosticsWriter:
    """Durably append compact campaign diagnostics without overwriting runs."""

    FILE_FIELDS = {
        "proto_campaign_geometry_summary.csv": [
            "round",
            "total",
            "true_class_cosine_micro",
            "nearest_wrong_cosine_micro",
            "proto_margin_micro",
            "true_class_cosine_macro",
            "nearest_wrong_cosine_macro",
            "proto_margin_macro",
            "within_class_dispersion_macro",
            "centroid_pair_cosine_mean",
            "centroid_pair_cosine_max",
            "inter_proto_cosine_mean",
            "inter_proto_cosine_min",
            "inter_proto_cosine_max",
            "classifier_weight_proto_cosine_mean",
            "classifier_weight_proto_compatible",
        ],
        "proto_campaign_geometry_classwise.csv": [
            "round",
            "class_id",
            "samples",
            "true_class_cosine",
            "nearest_wrong_cosine",
            "proto_margin",
            "within_class_feature_dispersion",
            "centroid_nearest_wrong_cosine",
            "centroid_mean_other_cosine",
            "classifier_weight_proto_cosine",
        ],
        "proto_campaign_prototype_state.csv": [
            "round",
            "class_id",
            "raw_round_support",
            "effective_count",
            "age",
            "reliability",
            "round_dispersion",
            "prototype_drift_cosine",
        ],
        "proto_campaign_loss_decomposition.csv": [
            "round",
            "proto_arm",
            "main_logits_kind",
            "observations",
            "linear_ce_raw",
            "main_ce_raw",
            "main_ce_weighted",
            "current_teacher_lmd_raw",
            "current_teacher_lmd_weighted",
            "historical_teacher_lmd_raw",
            "historical_teacher_lmd_weighted",
            "alignment_raw",
            "alignment_weighted",
            "proxy_ce_raw",
            "proxy_weighted",
            "fusion_related_objective",
            "total_loss",
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
                raise FileExistsError(f"campaign diagnostic exists: {path}")
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if int(row["round"]) <= int(completed_round)
                ]
            self._rewrite(path, fields, rows)

    def write_geometry(self, round_number, geometry):
        summary = {"round": int(round_number)}
        summary.update(
            {
                key: _finite_or_blank(value)
                for key, value in geometry["summary"].items()
            }
        )
        summary["total"] = int(geometry["summary"]["total"])
        summary["classifier_weight_proto_compatible"] = int(
            geometry["summary"]["classifier_weight_proto_compatible"]
        )
        self._append("proto_campaign_geometry_summary.csv", [summary])

        rows = []
        for values in geometry["classwise"]:
            row = {"round": int(round_number)}
            for key, value in values.items():
                row[key] = (
                    int(value)
                    if key in {"class_id", "samples"}
                    else _finite_or_blank(value)
                )
            rows.append(row)
        self._append("proto_campaign_geometry_classwise.csv", rows)

    def write_state(
        self,
        round_number,
        raw_support,
        effective_count,
        age,
        reliability,
        dispersion,
        drift_cosine,
    ):
        rows = []
        for class_id in range(self.num_classes):
            rows.append(
                {
                    "round": int(round_number),
                    "class_id": class_id,
                    "raw_round_support": (
                        "" if raw_support is None else int(raw_support[class_id])
                    ),
                    "effective_count": _finite_or_blank(
                        effective_count[class_id]
                    ),
                    "age": int(age[class_id]),
                    "reliability": _finite_or_blank(reliability[class_id]),
                    "round_dispersion": _finite_or_blank(
                        dispersion[class_id]
                    ),
                    "prototype_drift_cosine": _finite_or_blank(
                        drift_cosine[class_id]
                    ),
                }
            )
        self._append("proto_campaign_prototype_state.csv", rows)

    def write_losses(self, round_number, proto_arm, losses):
        if losses is None:
            return
        row = {
            "round": int(round_number),
            "proto_arm": str(proto_arm),
            "main_logits_kind": (
                "fused"
                if proto_arm in {
                    "full",
                    "wo_reliability",
                    "dr_v1",
                    "forward_no_r_jacobian_r",
                    "forward_r_jacobian_no_r",
                }
                else "linear"
            ),
        }
        for key in self.FILE_FIELDS["proto_campaign_loss_decomposition.csv"]:
            if key in row or key in {"round", "proto_arm", "main_logits_kind"}:
                continue
            value = losses.get(f"loss_{key}", float("nan"))
            row[key] = (
                int(value)
                if key == "observations" and math.isfinite(float(value))
                else _finite_or_blank(value)
            )
        self._append("proto_campaign_loss_decomposition.csv", [row])

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
