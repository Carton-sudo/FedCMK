"""Shared prototype-guided client optimization."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.BaseClientTrainer import BaseClientTrainer
from algorithms.fedcmk._shared.model_split import decouple_fedavgnet

__all__ = ["CountAgeClientTrainer"]


def _config_value(config, key, default):
    """Read normal objects and ConfLoader dictionaries with a real default."""
    try:
        return getattr(config, key)
    except (AttributeError, KeyError):
        return default


class CountAgeClientTrainer(BaseClientTrainer):
    """Client behavior selected by the wrapper class' ``VARIANT`` value."""

    VARIANT = None
    HISTORY_ENABLED = True

    def __init__(self, criterion, model, algo_params, local_epochs, device, num_classes):
        super().__init__(algo_params, model, local_epochs, device, num_classes)
        if self.VARIANT not in {
            "v221",
            "v222",
            "v223",
            "v224",
            "v224_vacant_ema",
        }:
            raise ValueError(f"Unsupported ProtoV2.2 variant: {self.VARIANT}")

        self.body, self.g_head_template = decouple_fedavgnet(model)
        self.body.to(device)
        self.g_head_template.to(device)
        self.linear_head = copy.deepcopy(self.g_head_template).to(device)

        self.num_classes = num_classes
        self.tau = algo_params.tau
        self.beta = algo_params.beta
        self.gamma = _config_value(algo_params, "gamma", 0.5)
        self.history_enabled = bool(self.HISTORY_ENABLED)
        if not self.history_enabled and float(self.gamma) != 0.0:
            raise ValueError(
                "history-free clients require gamma=0.0; disabling the "
                "historical loss must be explicit in the frozen config"
            )
        self.mu = _config_value(algo_params, "mu", 1.0)
        self.kappa = _config_value(algo_params, "kappa", 30.0)
        self.eta_max = _config_value(algo_params, "eta_max", 1.0)
        self.proto_temperature = _config_value(
            algo_params, "proto_temperature", 0.1
        )
        self.min_proto_samples = _config_value(
            algo_params, "min_proto_samples", 5
        )
        self.local_test_max_samples = _config_value(
            algo_params, "local_test_max_samples", 1000
        )
        self.proto_arm = _config_value(algo_params, "proto_arm", "full")
        if self.proto_arm not in {
            "noproto",
            "protoreg_only",
            "full",
            "wo_reliability",
            "dr_v1",
            "forward_no_r_jacobian_r",
            "forward_r_jacobian_no_r",
        }:
            raise ValueError(f"Unsupported Proto mechanism arm: {self.proto_arm}")
        self.proto_regularization_enabled = self.proto_arm != "noproto"
        self.proto_prediction_enabled = self.proto_arm in {
            "full",
            "wo_reliability",
            "dr_v1",
            "forward_no_r_jacobian_r",
            "forward_r_jacobian_no_r",
        }
        # Decision reliability is role-specific.  Full keeps the historical
        # behavior; wo_reliability removes it from both student and teachers;
        # DR-v1 removes only the student final-decision edge while retaining
        # reliability-aware teacher knowledge targets.
        self.student_decision_reliability_enabled = self.proto_arm not in {
            "wo_reliability",
            "dr_v1",
            "forward_no_r_jacobian_r",
        }
        self.teacher_decision_reliability_enabled = (
            self.proto_arm != "wo_reliability"
        )
        # Compatibility alias for external tests/legacy callers.  New code
        # dispatches student and teacher roles explicitly.
        self.reliability_in_eta = (
            self.student_decision_reliability_enabled
        )
        self.student_training_fusion_mode = {
            "forward_no_r_jacobian_r": "forward_no_r_jacobian_r",
            "forward_r_jacobian_no_r": "forward_r_jacobian_no_r",
        }.get(
            self.proto_arm,
            "r" if self.student_decision_reliability_enabled else "no_r",
        )
        # Server inference follows the candidate's forward value, not its
        # training-only surrogate Jacobian.  This keeps train-forward and
        # inference classifier formulas identical.
        self.student_evaluation_fusion_mode = {
            "forward_no_r_jacobian_r": "no_r",
            "forward_r_jacobian_no_r": "r",
        }.get(
            self.proto_arm,
            "r" if self.student_decision_reliability_enabled else "no_r",
        )
        self.campaign_diagnostic_enabled = bool(
            _config_value(algo_params, "campaign_diagnostic_enabled", False)
        )
        # ---- FedSAP proxy separation ----
        self.proxy_enabled = _config_value(
            algo_params, "proxy_enabled", False
        )
        self.proxy_scale = _config_value(algo_params, "proxy_scale", 32)
        self.proxy_weight = _config_value(algo_params, "proxy_weight", 0.5)
        self.proxy_loss_start_round = _config_value(
            algo_params, "proxy_loss_start_round", 20
        )
        self.proxy_loss_ramp_rounds = _config_value(
            algo_params, "proxy_loss_ramp_rounds", 80
        )

        feature_dim = self.g_head_template.in_features
        self.projector = None
        prototype_dim = feature_dim
        if self.VARIANT == "v223":
            prototype_dim = _config_value(
                algo_params, "proto_projection_dim", 128
            )
            self.projector = nn.Linear(
                feature_dim, prototype_dim, bias=False
            ).to(device)

        self.prototypes = F.normalize(
            torch.randn(num_classes, prototype_dim), dim=1
        ).to(device)
        self.proto_reliability = torch.zeros(num_classes, device=device)
        self.scaling = nn.Parameter(torch.tensor([10.0], device=device))

        self.dg_body = None
        self.dg_linear_head = None
        self.dg_projector = None
        self.dg_prototypes = None
        self.dg_reliability = None
        self.hist_body = None
        self.hist_linear_head = None
        self.hist_projector = None
        self.hist_prototypes = None
        self.hist_reliability = None

        self.criterion = criterion
        self.ce_loss_fn = nn.CrossEntropyLoss()

    @property
    def task_gate(self):
        return self.num_classes / (self.num_classes + self.kappa)

    def download_global(self, server_weights, server_optimizer):
        pass

    def download_v22(
        self,
        body_state,
        linear_head_state,
        projector_state,
        prototypes,
        proto_reliability,
        hist_body_state,
        hist_linear_head_state,
        hist_projector_state,
        hist_prototypes,
        hist_reliability,
        server_optimizer,
    ):
        self.body.load_state_dict(body_state)
        if linear_head_state is not None:
            self.linear_head.load_state_dict(linear_head_state)
        if self.projector is not None and projector_state is not None:
            self.projector.load_state_dict(projector_state)
        self.prototypes = prototypes.clone().to(self.device)
        self.proto_reliability = proto_reliability.clone().to(self.device)

        self.dg_body = self._frozen_module(self.body, body_state)
        if self.history_enabled:
            if hist_body_state is None:
                raise ValueError("history-enabled client requires hist_body_state")
            self.hist_body = self._frozen_module(self.body, hist_body_state)
        elif any(
            value is not None
            for value in (
                hist_body_state,
                hist_linear_head_state,
                hist_projector_state,
                hist_prototypes,
                hist_reliability,
            )
        ):
            raise ValueError("history-free client received historical state")

        if self.VARIANT != "v221":
            self.dg_linear_head = self._frozen_module(
                self.linear_head, linear_head_state
            )
            if self.history_enabled:
                self.hist_linear_head = self._frozen_module(
                    self.linear_head, hist_linear_head_state
                )
        if self.projector is not None:
            self.dg_projector = self._frozen_module(
                self.projector, projector_state
            )
            if self.history_enabled:
                self.hist_projector = self._frozen_module(
                    self.projector, hist_projector_state
                )

        self.dg_prototypes = prototypes.clone().to(self.device)
        self.dg_reliability = proto_reliability.clone().to(self.device)
        if self.history_enabled:
            self.hist_prototypes = hist_prototypes.clone().to(self.device)
            self.hist_reliability = hist_reliability.clone().to(self.device)

        trainable = list(self.body.parameters())
        if self.VARIANT == "v221":
            trainable.append(self.scaling)
        else:
            trainable.extend(self.linear_head.parameters())
            if self.projector is not None:
                trainable.extend(self.projector.parameters())

        param_group = server_optimizer["param_groups"][0] if server_optimizer else {}
        self.optimizer = torch.optim.SGD(
            trainable,
            lr=param_group.get("lr", 0.01),
            momentum=param_group.get("momentum", 0.9),
            weight_decay=param_group.get("weight_decay", 1e-5),
        )

    def _frozen_module(self, template, state):
        module = copy.deepcopy(template)
        module.load_state_dict(state)
        module.to(self.device)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
        return module

    def train(self, round_number, evaluate_metrics=False, proto_lambda=0.0):
        self.body.to(self.device)
        self.body.train()
        self.current_round = round_number
        if self.VARIANT == "v221":
            diagnostics = self._train_proto_only(proto_lambda)
        else:
            self.linear_head.to(self.device)
            self.linear_head.train()
            if self.projector is not None:
                self.projector.to(self.device)
                self.projector.train()
            diagnostics = self._train_hybrid(proto_lambda)

        local_prototypes, counts, variances = self._estimate_prototypes()
        local_results = (
            self._get_local_stats()
            if evaluate_metrics and self.local_metrics_enabled
            else {}
        )
        return (
            local_results,
            self.datasize,
            local_prototypes,
            counts,
            variances,
            diagnostics,
        )

    def _proxy_lambda(self):
        """FedSAP: linear ramp from round 20 to 100, max 1.0. Only active if proxy_enabled."""
        if not self.proxy_enabled:
            return 0.0
        r = self.current_round + 1  # round_number is 1-based
        if r < self.proxy_loss_start_round:
            return 0.0
        if r >= self.proxy_loss_start_round + self.proxy_loss_ramp_rounds:
            return 1.0
        return (r - self.proxy_loss_start_round) / self.proxy_loss_ramp_rounds

    def _train_proto_only(self, proto_lambda):
        prototypes = self.prototypes.to(self.device)
        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()
                data, targets = data.to(self.device), targets.to(self.device)

                features = self.body(data)
                norm_features = F.normalize(features, dim=1)
                norm_prototypes = F.normalize(prototypes, dim=1)
                logits = self.scaling * (norm_features @ norm_prototypes.T)

                with torch.no_grad():
                    global_features = F.normalize(self.dg_body(data), dim=1)
                    global_logits = self.scaling * (
                        global_features @ F.normalize(self.dg_prototypes, dim=1).T
                    )
                    history_features = F.normalize(self.hist_body(data), dim=1)
                    history_logits = self.scaling * (
                        history_features
                        @ F.normalize(self.hist_prototypes, dim=1).T
                    )

                base_loss = (
                    self.ce_loss_fn(logits, targets)
                    + self.beta
                    * self._lmd(logits, global_logits, targets, self.major_labels)
                    + self.gamma
                    * self._lmd(logits, history_logits, targets, self.major_labels)
                )
                alignment = 1.0 - F.cosine_similarity(
                    norm_features, norm_prototypes[targets], dim=1
                )
                loss = base_loss + proto_lambda * alignment.mean()
                loss.backward()
                self.optimizer.step()

        return {
            "task_gate": 1.0,
            "sample_gate": 1.0,
            "class_gate": 1.0,
            "eta": 1.0,
        }

    def _train_hybrid(self, proto_lambda):
        self._reset_extra_hybrid_diagnostics()
        # These are reporting-only statistics.  Accumulate on device and
        # transfer once per client instead of forcing three device
        # synchronizations for every local batch.
        # float32 is supported by CPU/CUDA/MPS.  The values never enter the
        # objective or gradient graph; they are reporting-only aggregates.
        gate_sums = torch.zeros(3, device=self.device, dtype=torch.float32)
        observations = 0
        loss_sums = {
            "linear_ce_raw": 0.0,
            "main_ce_raw": 0.0,
            "main_ce_weighted": 0.0,
            "current_teacher_lmd_raw": 0.0,
            "current_teacher_lmd_weighted": 0.0,
            "historical_teacher_lmd_raw": 0.0,
            "historical_teacher_lmd_weighted": 0.0,
            "alignment_raw": 0.0,
            "alignment_weighted": 0.0,
            "proxy_ce_raw": 0.0,
            "proxy_weighted": 0.0,
            "fusion_related_objective": 0.0,
            "total_loss": 0.0,
        }

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()
                data, targets = data.to(self.device), targets.to(self.device)

                student = self._hybrid_forward(
                    data,
                    self.body,
                    self.linear_head,
                    self.projector,
                    self.prototypes,
                    self.proto_reliability,
                    fusion_mode=self.student_training_fusion_mode,
                )
                with torch.no_grad():
                    global_teacher = self._hybrid_forward(
                        data,
                        self.dg_body,
                        self.dg_linear_head,
                        self.dg_projector,
                        self.dg_prototypes,
                        self.dg_reliability,
                        use_reliability_in_eta=(
                            self.teacher_decision_reliability_enabled
                        ),
                    )
                    history_teacher = (
                        self._hybrid_forward(
                            data,
                            self.hist_body,
                            self.hist_linear_head,
                            self.hist_projector,
                            self.hist_prototypes,
                            self.hist_reliability,
                            use_reliability_in_eta=(
                                self.teacher_decision_reliability_enabled
                            ),
                        )
                        if self.history_enabled
                        else None
                    )

                linear_ce_raw = self.ce_loss_fn(
                    student["linear_logits"], targets
                )
                main_ce_raw = self.ce_loss_fn(student["fused_logits"], targets)
                current_lmd_raw = self._lmd(
                    student["fused_logits"],
                    global_teacher["fused_logits"],
                    targets,
                    self.major_labels,
                )
                historical_lmd_raw = (
                    self._lmd(
                        student["fused_logits"],
                        history_teacher["fused_logits"],
                        targets,
                        self.major_labels,
                    )
                    if self.history_enabled
                    else linear_ce_raw.detach() * 0.0
                )
                main_ce_weighted = self.mu * main_ce_raw
                current_lmd_weighted = self.beta * current_lmd_raw
                historical_lmd_weighted = (
                    self.gamma * historical_lmd_raw
                    if self.history_enabled
                    else linear_ce_raw.detach() * 0.0
                )
                base_loss = (
                    linear_ce_raw
                    + main_ce_weighted
                    + current_lmd_weighted
                    + historical_lmd_weighted
                )

                target_prototypes = F.normalize(
                    self.prototypes, dim=1
                )[targets]
                alignment = 1.0 - F.cosine_similarity(
                    student["proto_features"], target_prototypes, dim=1
                )
                if self.VARIANT == "v222":
                    proto_gate = torch.full_like(alignment, self.task_gate)
                else:
                    proto_gate = (
                        self.task_gate
                        * student["sample_gate"]
                        * self.proto_reliability[targets]
                    )
                alignment_raw = (proto_gate * alignment).mean()
                if self.proto_regularization_enabled:
                    alignment_weighted = proto_lambda * alignment_raw
                    loss = base_loss + alignment_weighted
                else:
                    alignment_weighted = alignment_raw.detach() * 0.0
                    loss = base_loss
                # Variant-specific regularizers are strictly additive: the
                # original ProtoV2.2.4 objective above remains unchanged.
                loss = loss + self._extra_hybrid_loss(
                    student,
                    global_teacher,
                    history_teacher,
                    targets,
                )

                # ---- FedSAP proxy separation loss ----
                proxy_lambda_val = (
                    self._proxy_lambda()
                    if self.proto_regularization_enabled
                    else 0.0
                )
                proxy_ce = linear_ce_raw.detach() * 0.0
                proxy_weighted = linear_ce_raw.detach() * 0.0
                if proxy_lambda_val > 0:
                    norm_protos = F.normalize(self.prototypes, dim=1)
                    proxy_logits = self.proxy_scale * (
                        student["proto_features"] @ norm_protos.T
                    )
                    proxy_ce = self.ce_loss_fn(proxy_logits, targets)
                    proxy_weighted = (
                        proxy_lambda_val * self.proxy_weight * proxy_ce
                    )
                    loss = loss + proxy_weighted

                loss.backward()
                self.optimizer.step()

                batch_size = targets.size(0)
                gate_sums[0] += student["sample_gate"].detach().sum()
                target_reliability = self.proto_reliability[targets]
                gate_sums[1] += target_reliability.detach().sum()
                gate_sums[2] += (
                    student["eta"].detach().mean(dim=1).sum()
                )
                observations += batch_size
                if self.campaign_diagnostic_enabled:
                    weighted_values = {
                        "linear_ce_raw": linear_ce_raw,
                        "main_ce_raw": main_ce_raw,
                        "main_ce_weighted": main_ce_weighted,
                        "current_teacher_lmd_raw": current_lmd_raw,
                        "current_teacher_lmd_weighted": current_lmd_weighted,
                        "historical_teacher_lmd_raw": historical_lmd_raw,
                        "historical_teacher_lmd_weighted": (
                            historical_lmd_weighted
                        ),
                        "alignment_raw": alignment_raw,
                        "alignment_weighted": alignment_weighted,
                        "proxy_ce_raw": proxy_ce,
                        "proxy_weighted": proxy_weighted,
                        "fusion_related_objective": (
                            main_ce_weighted
                            + current_lmd_weighted
                            + historical_lmd_weighted
                            if self.proto_prediction_enabled
                            else linear_ce_raw.detach() * 0.0
                        ),
                        "total_loss": loss,
                    }
                    for key, value in weighted_values.items():
                        loss_sums[key] += (
                            float(value.detach().item()) * batch_size
                        )

        denominator = max(observations, 1)
        gate_values = gate_sums.detach().cpu().tolist()
        diagnostics = {
            "task_gate": float(self.task_gate),
            "sample_gate": gate_values[0] / denominator,
            "class_gate": gate_values[1] / denominator,
            "eta": gate_values[2] / denominator,
        }
        if self.campaign_diagnostic_enabled:
            diagnostics["loss_observations"] = int(observations)
            diagnostics.update(
                {
                    f"loss_{key}": value / denominator
                    for key, value in loss_sums.items()
                }
            )
        diagnostics.update(self._extra_hybrid_diagnostics())
        return diagnostics

    def _reset_extra_hybrid_diagnostics(self):
        """Reset optional diagnostics for a derived hybrid variant."""

    def _extra_hybrid_loss(
        self,
        student,
        global_teacher,
        history_teacher,
        targets,
    ):
        """Return an additive loss without changing the shared objective."""
        return 0.0

    def _extra_hybrid_diagnostics(self):
        """Return optional round diagnostics for a derived hybrid variant."""
        return {}

    def _hybrid_forward(
        self,
        data,
        body,
        linear_head,
        projector,
        prototypes,
        reliability,
        use_reliability_in_eta=None,
        fusion_mode=None,
    ):
        features = body(data)
        linear_logits = linear_head(features)
        projected = projector(features) if projector is not None else features
        proto_features = F.normalize(projected, dim=1)
        proto_logits = (
            proto_features @ F.normalize(prototypes, dim=1).T
        ) / self.proto_temperature

        if self.VARIANT == "v222":
            sample_gate = torch.ones(
                features.size(0), device=features.device
            )
            eta = torch.full_like(
                proto_logits, self.eta_max * self.task_gate
            )
            eta_reliable = eta
            eta_no_reliability = eta
        else:
            probabilities = F.softmax(linear_logits, dim=1)
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(dim=1)
            sample_gate = (entropy / math.log(self.num_classes)).clamp(0.0, 1.0)
            if use_reliability_in_eta is None:
                use_reliability_in_eta = getattr(
                    self,
                    "student_decision_reliability_enabled",
                    getattr(self, "reliability_in_eta", True),
                )
            eta_no_reliability = (
                self.eta_max
                * self.task_gate
                * sample_gate.unsqueeze(1)
                * torch.ones_like(reliability).unsqueeze(0)
            ).expand(-1, proto_logits.size(1))
            eta_reliable = eta_no_reliability * reliability.unsqueeze(0)
            eta = eta_reliable if use_reliability_in_eta else eta_no_reliability

        if not getattr(self, "proto_prediction_enabled", True):
            eta = torch.zeros_like(proto_logits)
            eta_reliable = torch.zeros_like(proto_logits)
            eta_no_reliability = torch.zeros_like(proto_logits)

        if fusion_mode is None:
            fusion_mode = "r" if use_reliability_in_eta else "no_r"

        # Ordinary R and No-R forwards only need one fused expression.  The
        # Stage-5 surrogate-Jacobian modes intentionally retain both paths.
        if fusion_mode == "r":
            eta = eta_reliable
            fused_logits = linear_logits + eta * proto_logits
        elif fusion_mode == "no_r":
            eta = eta_no_reliability
            fused_logits = linear_logits + eta * proto_logits
        elif fusion_mode in {
            "forward_no_r_jacobian_r",
            "forward_r_jacobian_no_r",
        }:
            reliable_fused_logits = (
                linear_logits + eta_reliable * proto_logits
            )
            no_reliability_fused_logits = (
                linear_logits + eta_no_reliability * proto_logits
            )
            fused_logits = self._compose_fused_logits(
                reliable_fused_logits,
                no_reliability_fused_logits,
                fusion_mode,
            )
            eta = (
                eta_no_reliability
                if fusion_mode == "forward_no_r_jacobian_r"
                else eta_reliable
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        if fusion_mode in {"r", "forward_r_jacobian_no_r"}:
            eta = eta_reliable

        return {
            "linear_logits": linear_logits,
            "proto_logits": proto_logits,
            "fused_logits": fused_logits,
            "main_logits": fused_logits,
            "proto_features": proto_features,
            "proto_cosine_logits": proto_logits * self.proto_temperature,
            "sample_gate": sample_gate,
            "eta": eta,
            "fusion_mode": fusion_mode,
        }

    @staticmethod
    def _compose_fused_logits(
        reliable_fused_logits,
        no_reliability_fused_logits,
        fusion_mode,
    ):
        """Select a forward value and, for Stage 5A, a surrogate Jacobian.

        The stop-gradient modes alter only the Jacobian of the student fused
        logits.  They do not assert equality of the downstream loss gradient:
        dL/dz is evaluated at the selected forward logits.
        """
        if fusion_mode == "r":
            return reliable_fused_logits
        if fusion_mode == "no_r":
            return no_reliability_fused_logits
        if fusion_mode == "forward_no_r_jacobian_r":
            return reliable_fused_logits + (
                no_reliability_fused_logits - reliable_fused_logits
            ).detach()
        if fusion_mode == "forward_r_jacobian_no_r":
            return no_reliability_fused_logits + (
                reliable_fused_logits - no_reliability_fused_logits
            ).detach()
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

    @torch.no_grad()
    def _estimate_prototypes(self):
        prototype_dim = self.prototypes.size(1)
        feature_sums = torch.zeros(
            self.num_classes, prototype_dim, device=self.device
        )
        raw_counts = torch.zeros(self.num_classes, device=self.device)

        self.body.eval()
        if self.projector is not None:
            self.projector.eval()
        for data, targets in self.trainloader:
            data, targets = data.to(self.device), targets.to(self.device)
            body_features = self.body(data)
            projected = (
                self.projector(body_features)
                if self.projector is not None
                else body_features
            )
            normalized = F.normalize(projected, dim=1)
            for class_idx in range(self.num_classes):
                mask = targets == class_idx
                if mask.any():
                    feature_sums[class_idx] += normalized[mask].sum(dim=0)
                    raw_counts[class_idx] += mask.sum()

        reliable_prototypes = torch.zeros_like(feature_sums)
        reliable_counts = torch.zeros_like(raw_counts)
        variances = torch.zeros_like(raw_counts)
        for class_idx in range(self.num_classes):
            count = raw_counts[class_idx]
            if count >= self.min_proto_samples:
                vector_sum = feature_sums[class_idx]
                reliable_prototypes[class_idx] = F.normalize(
                    vector_sum.unsqueeze(0), dim=1
                ).squeeze(0)
                reliable_counts[class_idx] = count
                variances[class_idx] = torch.clamp(
                    2.0 - 2.0 * vector_sum.norm() / count,
                    min=0.0,
                )

        return (
            reliable_prototypes.cpu(),
            reliable_counts.cpu(),
            variances.cpu(),
        )

    def _lmd(self, student_logits, teacher_logits, targets, major_labels):
        batch_size = student_logits.size(0)
        ground_truth_mask = torch.zeros_like(student_logits).scatter_(
            1, targets.reshape(-1, 1), 1
        ).bool()
        major_label_mask = torch.zeros_like(student_logits).scatter_(
            1, major_labels.repeat(batch_size, 1), 1
        ).bool()
        teacher_probabilities = F.softmax(
            teacher_logits / self.tau
            - 1000 * major_label_mask
            - 1000 * ground_truth_mask,
            dim=1,
        )
        student_log_probabilities = F.log_softmax(
            student_logits / self.tau - 1000 * ground_truth_mask,
            dim=1,
        )
        return (
            nn.KLDivLoss(reduction="batchmean")(
                student_log_probabilities, teacher_probabilities
            )
            * (self.tau**2)
        )

    def _get_local_stats(self):
        self.body.eval()
        if self.VARIANT != "v221":
            self.linear_head.eval()
        if self.projector is not None:
            self.projector.eval()

        def evaluate(dataloader, max_samples=None):
            correct = 0
            total = 0
            for data, targets in dataloader:
                if max_samples is not None:
                    remaining = max_samples - total
                    if remaining <= 0:
                        break
                    data = data[:remaining]
                    targets = targets[:remaining]
                data, targets = data.to(self.device), targets.to(self.device)

                if self.VARIANT == "v221":
                    features = F.normalize(self.body(data), dim=1)
                    logits = self.scaling * (
                        features @ F.normalize(self.prototypes, dim=1).T
                    )
                else:
                    logits = self._hybrid_forward(
                        data,
                        self.body,
                        self.linear_head,
                        self.projector,
                        self.prototypes,
                        self.proto_reliability,
                    )["fused_logits"]
                correct += (logits.argmax(dim=1) == targets).sum().item()
                total += targets.size(0)
            return correct / total if total else 0.0

        with torch.no_grad():
            return {
                "train_acc": evaluate(self.trainloader),
                "test_acc": evaluate(
                    self.testloader, self.local_test_max_samples
                ),
                "classwise_accuracy": None,
            }

    def upload_local(self):
        uploaded = {"body": copy.deepcopy(self.body.state_dict())}
        if self.VARIANT != "v221":
            uploaded["linear_head"] = copy.deepcopy(
                self.linear_head.state_dict()
            )
        if self.projector is not None:
            uploaded["projector"] = copy.deepcopy(
                self.projector.state_dict()
            )
        return uploaded

    def reset(self):
        # download_v22 always constructs the real client optimizer before the
        # next local train.  Avoid BaseClientTrainer's throw-away SGD(lr=0)
        # object while preserving the same cleared client state.
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = None
        self.dg_body = None
        self.dg_linear_head = None
        self.dg_projector = None
        self.dg_prototypes = None
        self.dg_reliability = None
        self.hist_body = None
        self.hist_linear_head = None
        self.hist_projector = None
        self.hist_prototypes = None
        self.hist_reliability = None
