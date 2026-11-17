"""Shared prototype-memory and server training operations."""

import copy
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from algorithms.BaseServer import BaseServer
from algorithms.fedcmk._shared.losses import (
    LMD_Hist_Loss,
    proto_lambda_at_round,
)
from algorithms.fedcmk._shared.model_split import decouple_fedavgnet
from algorithms.fedcmk._shared.t2_diagnostics import (
    T2DiagnosticsWriter,
    T2EvaluationAccumulator,
)
from algorithms.fedcmk._shared.proto_campaign_diagnostics import (
    ProtoCampaignDiagnosticsWriter,
    ProtoGeometryAccumulator,
)
from algorithms.fedcmk._shared.sample_prediction_diagnostics import (
    SamplePredictionAccumulator,
    SamplePredictionWriter,
)

__all__ = ["CountAgeServer"]


def _config_value(config, key, default):
    """Read normal objects and ConfLoader dictionaries with a real default."""
    try:
        return getattr(config, key)
    except (AttributeError, KeyError):
        return default


class CountAgeServer(BaseServer):
    VARIANT = None
    DISPLAY_NAME = None
    CLIENT_CLASS = None
    HISTORY_ENABLED = True

    def __init__(
        self,
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler=None,
        **kwargs,
    ):
        diagnostic_dir = kwargs.pop("diagnostic_dir", None)
        super().__init__(
            algo_params,
            model,
            data_distributed,
            optimizer,
            scheduler,
            **kwargs,
        )
        if self.VARIANT not in {
            "v221",
            "v222",
            "v223",
            "v224",
            "v224_vacant_ema",
        }:
            raise ValueError(f"Unsupported ProtoV2.2 variant: {self.VARIANT}")
        if self.CLIENT_CLASS is None:
            raise ValueError("ProtoV2.2 wrapper must define CLIENT_CLASS")

        self.t2_diagnostics_enabled = bool(
            _config_value(algo_params, "diagnostic_enabled", False)
        )
        if self.t2_diagnostics_enabled and self.VARIANT != "v224":
            raise ValueError(
                "T2 diagnostics are frozen for formal ProtoV2.2.4 only"
            )
        if diagnostic_dir is None and self.output_dir is not None:
            diagnostic_dir = self.output_dir / "diagnostics"
        self.t2_diagnostics = (
            T2DiagnosticsWriter(
                Path(diagnostic_dir),
                self.num_classes,
                overwrite=self.overwrite_output,
            )
            if self.t2_diagnostics_enabled and diagnostic_dir is not None
            else None
        )
        if self.t2_diagnostics_enabled and self.t2_diagnostics is None:
            raise ValueError(
                "diagnostic_enabled=true requires output_dir or diagnostic_dir"
            )

        self.campaign_diagnostics_enabled = bool(
            _config_value(
                algo_params, "campaign_diagnostic_enabled", False
            )
        )
        if self.campaign_diagnostics_enabled and self.VARIANT != "v224":
            raise ValueError(
                "Proto campaign diagnostics are frozen for "
                "formal ProtoV2.2.4 only"
            )
        self.campaign_diagnostics = (
            ProtoCampaignDiagnosticsWriter(
                Path(diagnostic_dir),
                self.num_classes,
                overwrite=self.overwrite_output,
            )
            if self.campaign_diagnostics_enabled
            and diagnostic_dir is not None
            else None
        )
        if (
            self.campaign_diagnostics_enabled
            and self.campaign_diagnostics is None
        ):
            raise ValueError(
                "campaign_diagnostic_enabled=true requires output_dir "
                "or diagnostic_dir"
            )

        self.sample_prediction_logging_enabled = bool(
            _config_value(
                algo_params, "sample_prediction_logging_enabled", False
            )
        )
        self.sample_prediction_start_round = int(
            _config_value(algo_params, "sample_prediction_start_round", 150)
        )
        if self.sample_prediction_logging_enabled and self.VARIANT != "v224":
            raise ValueError(
                "per-sample prediction logging is supported only for "
                "ProtoV2.2.4"
            )
        if self.sample_prediction_logging_enabled:
            from torch.utils.data import SequentialSampler

            if not isinstance(self.testloader.sampler, SequentialSampler):
                raise ValueError(
                    "per-sample prediction logging requires an unshuffled "
                    "global test loader"
                )
            if bool(getattr(self.testloader, "drop_last", False)):
                raise ValueError(
                    "per-sample prediction logging requires drop_last=False"
                )
        self.sample_prediction_writer = (
            SamplePredictionWriter(
                Path(diagnostic_dir),
                start_round=self.sample_prediction_start_round,
                overwrite=self.overwrite_output,
            )
            if self.sample_prediction_logging_enabled
            and diagnostic_dir is not None
            else None
        )
        if (
            self.sample_prediction_logging_enabled
            and self.sample_prediction_writer is None
        ):
            raise ValueError(
                "sample_prediction_logging_enabled=true requires output_dir "
                "or diagnostic_dir"
            )
        self._campaign_round_dispersion = torch.full(
            (self.num_classes,), float("nan"), device=self.device
        )
        self._campaign_proto_drift = torch.full(
            (self.num_classes,), float("nan"), device=self.device
        )

        self.history_alpha = _config_value(
            algo_params, "history_alpha", 0.3
        )
        self.history_enabled = bool(self.HISTORY_ENABLED)
        configured_gamma = float(_config_value(algo_params, "gamma", 0.5))
        if not self.history_enabled and configured_gamma != 0.0:
            raise ValueError(
                "history-free servers require gamma=0.0; the config must "
                "record the removed historical-teacher objective explicitly"
            )
        self.eval_interval = _config_value(algo_params, "eval_interval", 3)
        self.rho = _config_value(algo_params, "rho", 0.98)
        self.tau_n = _config_value(algo_params, "tau_n", 5.0)
        self.tau_v = _config_value(algo_params, "tau_v", 1.0)
        self.tau_R = _config_value(algo_params, "tau_R", 10.0)
        self.age_gamma = _config_value(algo_params, "age_gamma", 0.01)
        self.alpha_min = _config_value(algo_params, "alpha_min", 0.02)
        self.alpha_max = _config_value(algo_params, "alpha_max", 0.5)
        self.eps = _config_value(algo_params, "eps", 1e-12)

        self.proto_lambda_base = _config_value(
            algo_params, "proto_lambda_base", 1.0
        )
        self.proto_lambda_scale = _config_value(
            algo_params, "proto_lambda_scale", 0.25
        )
        self.proto_loss_start_round = _config_value(
            algo_params, "proto_loss_start_round", 100
        )
        self.proto_loss_ramp_rounds = _config_value(
            algo_params, "proto_loss_ramp_rounds", 100
        )

        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be > 0")
        if not 0 < self.rho < 1:
            raise ValueError("rho must be between 0 and 1")
        if self.tau_n <= 0 or self.tau_v <= 0 or self.tau_R <= 0:
            raise ValueError("tau_n, tau_v, and tau_R must be > 0")
        if not 0 <= self.alpha_min <= self.alpha_max <= 1:
            raise ValueError("alpha bounds must satisfy 0 <= min <= max <= 1")

        model_copy = copy.deepcopy(model)
        self.body, source_head = decouple_fedavgnet(model_copy)
        self.body.to(self.device)
        self.linear_head = copy.deepcopy(source_head).to(self.device)

        feature_dim = source_head.in_features
        self.projector = None
        prototype_dim = feature_dim
        if self.VARIANT == "v223":
            prototype_dim = _config_value(
                algo_params, "proto_projection_dim", 128
            )
            self.projector = torch.nn.Linear(
                feature_dim, prototype_dim, bias=False
            ).to(self.device)

        self.prototypes = F.normalize(
            torch.randn(self.num_classes, prototype_dim), dim=1
        ).to(self.device)
        self.scaling = torch.nn.Parameter(
            torch.tensor([10.0], device=self.device)
        )

        self.effective_count = torch.zeros(
            self.num_classes, device=self.device
        )
        self.prototype_age = torch.zeros(
            self.num_classes, dtype=torch.long, device=self.device
        )
        self.proto_reliability = torch.zeros(
            self.num_classes, device=self.device
        )
        self.alpha_update = torch.zeros(
            self.num_classes, device=self.device
        )

        self.hist_body = None
        self.hist_linear_head = None
        self.hist_projector = None
        self.hist_prototypes = None
        self.hist_reliability = None
        if self.history_enabled:
            self.hist_body = copy.deepcopy(self.body)
            self.hist_linear_head = copy.deepcopy(self.linear_head)
            self.hist_projector = (
                copy.deepcopy(self.projector)
                if self.projector is not None
                else None
            )
            self.hist_prototypes = self.prototypes.clone()
            self.hist_reliability = self.proto_reliability.clone()
            self._freeze(self.hist_body)
            self._freeze(self.hist_linear_head)
            if self.hist_projector is not None:
                self._freeze(self.hist_projector)

        criterion = LMD_Hist_Loss(
            self.num_classes,
            algo_params.tau,
            algo_params.beta,
            _config_value(algo_params, "gamma", 0.5),
        )
        self.client = self.CLIENT_CLASS(
            criterion,
            model=copy.deepcopy(model),
            algo_params=algo_params,
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )
        decision_reliability_enabled = bool(
            self.client.student_decision_reliability_enabled
        )
        self.run_metadata.update(
            {
                "proto_arm": self.client.proto_arm,
                "decision_gate_class_reliability_enabled": (
                    decision_reliability_enabled
                ),
                "teacher_decision_gate_class_reliability_enabled": bool(
                    self.client.teacher_decision_reliability_enabled
                ),
                "reliability_role": "prototype_knowledge_quality_control",
                "student_training_fusion_mode": (
                    self.client.student_training_fusion_mode
                ),
                "student_evaluation_fusion_mode": (
                    self.client.student_evaluation_fusion_mode
                ),
                "sample_prediction_logging_enabled": (
                    self.sample_prediction_logging_enabled
                ),
                "sample_prediction_start_round": (
                    self.sample_prediction_start_round
                ),
                "sample_id_semantics": (
                    "global_test_dataset_sequential_index"
                    if self.sample_prediction_logging_enabled
                    else None
                ),
                "historical_ema_teacher_present": self.history_enabled,
                "prototype_components_present": True,
                "evidence_grade": _config_value(
                    algo_params,
                    "evidence_grade",
                    None,
                ),
                "reference_run_id": _config_value(
                    algo_params,
                    "reference_run_id",
                    None,
                ),
            }
        )
        print(
            "[Decision Gate] class_reliability_enabled="
            f"{decision_reliability_enabled}"
        )
        # BaseServer creates meta.json before the algorithm-specific client is
        # available.  Refresh only this new run's metadata after the discrete
        # decision-gate identity has been resolved.
        self._write_meta()

        self.client_y_lst = {}
        for client_idx, client_data in self.data_distributed["local"].items():
            label_counts = defaultdict(lambda: 0)
            for _, labels in client_data["train"]:
                for label in labels:
                    label_counts[label.item()] += 1
            mean_count = np.mean(
                self.data_distributed["data_map"][client_idx]
            )
            major_labels = np.where(
                np.array(self.data_distributed["data_map"][client_idx])
                > mean_count
            )[0]
            self.client_y_lst[client_idx] = torch.tensor(
                major_labels, device=self.device
            )

        print(
            f"\n>>> {self.DISPLAY_NAME} Server initialized "
            f"(rho={self.rho}, alpha=[{self.alpha_min}, {self.alpha_max}], "
            f"eval_interval={self.eval_interval})...\n"
        )

    def _freeze(self, module):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    def run(self):
        self._print_start()
        completed_round = self._maybe_resume()
        self._prepare_t2_diagnostics(completed_round)
        self._prepare_campaign_diagnostics(completed_round)
        self._prepare_sample_prediction_diagnostics(completed_round)
        if completed_round == 0:
            eval_start = time.perf_counter()
            initial = self._evaluate_global(round_number=0)
            eval_sec = time.perf_counter() - eval_start
            t2_evaluation = initial.pop("_t2_diagnostics", None)
            campaign_geometry = initial.pop("_campaign_geometry", None)
            sample_predictions = initial.pop("_sample_predictions", None)
            self._store_evaluation(0, initial)
            print(
                f"[Round 0/{self.n_rounds}] "
                f"[Server Stat] Acc - {initial['main']:.4f} "
                f"({initial['main_correct']}/{initial['total']})"
            )
            if self.VARIANT != "v221":
                self._print_branch_accuracy(initial)
            wandb.log({"server_test_acc": initial["main"]}, step=0)
            self._append_metric_row(
                round_number=0,
                evaluated=True,
                evaluation=initial,
                learning_rate=self._current_learning_rate(),
                sampled_clients=[],
                train_sec=0.0,
                aggregate_sec=0.0,
                eval_sec=eval_sec,
                checkpoint_sec=0.0,
                round_total_sec=eval_sec,
                extra={"proto_lambda": 0.0},
            )
            self._commit_t2_diagnostics(0, [], t2_evaluation)
            self._commit_campaign_diagnostics(
                0,
                [],
                campaign_geometry,
                losses=None,
            )
            self._commit_sample_prediction_diagnostics(
                0, sample_predictions
            )
            print("-" * 50)

        for round_idx in range(completed_round, self.n_rounds):
            round_number = round_idx + 1
            start_time = time.perf_counter()
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            previous_states = self._history_source_states()
            evaluate_metrics = self._should_evaluate(round_number)
            proto_lambda = self._proto_lambda(round_number)
            learning_rate = self._current_learning_rate()

            train_start = time.perf_counter()
            (
                uploads,
                client_sizes,
                prototype_list,
                count_list,
                variance_list,
                round_results,
                gate_diagnostics,
            ) = self._clients_training(
                sampled_clients,
                round_number,
                evaluate_metrics,
                proto_lambda,
            )
            train_sec = time.perf_counter() - train_start

            aggregate_start = time.perf_counter()
            self._aggregate_models(uploads, client_sizes)
            self._aggregate_prototypes(
                prototype_list, count_list, variance_list
            )
            self._update_histories(previous_states)
            aggregate_sec = time.perf_counter() - aggregate_start

            self._step_scheduler()

            evaluation, eval_sec = self._finish_round(
                round_results,
                gate_diagnostics,
                round_number,
                start_time,
                evaluate_metrics,
                learning_rate,
                proto_lambda,
            )
            t2_evaluation = (
                evaluation.pop("_t2_diagnostics", None)
                if evaluation is not None
                else None
            )
            sample_predictions = (
                evaluation.pop("_sample_predictions", None)
                if evaluation is not None
                else None
            )
            pre_checkpoint_total_sec = time.perf_counter() - start_time
            self._append_metric_row(
                round_number=round_number,
                evaluated=evaluate_metrics,
                evaluation=evaluation,
                learning_rate=learning_rate,
                sampled_clients=sampled_clients,
                train_sec=train_sec,
                aggregate_sec=aggregate_sec,
                eval_sec=eval_sec,
                checkpoint_sec=0.0,
                round_total_sec=pre_checkpoint_total_sec,
                extra={"proto_lambda": repr(float(proto_lambda))},
            )
            self._commit_t2_diagnostics(
                round_number, sampled_clients, t2_evaluation
            )
            self._commit_sample_prediction_diagnostics(
                round_number, sample_predictions
            )
            campaign_geometry = (
                evaluation.pop("_campaign_geometry", None)
                if evaluation is not None
                else None
            )
            self._commit_campaign_diagnostics(
                round_number,
                sampled_clients,
                campaign_geometry,
                losses=gate_diagnostics,
            )
            checkpoint_start = time.perf_counter()
            checkpoint_written = self._checkpoint_if_due(round_number)
            checkpoint_sec = time.perf_counter() - checkpoint_start
            if checkpoint_written:
                self._update_metric_row(
                    round_number,
                    {
                        "checkpoint_sec": f"{checkpoint_sec:.9f}",
                        "round_total_sec": (
                            f"{time.perf_counter() - start_time:.9f}"
                        ),
                    },
                )

    def _clients_training(
        self,
        sampled_clients,
        round_number,
        evaluate_metrics,
        proto_lambda,
    ):
        uploads = []
        client_sizes = []
        prototype_list = []
        count_list = []
        variance_list = []
        round_results = {}
        diagnostic_list = []

        body_state = self.body.state_dict()
        linear_state = (
            self.linear_head.state_dict()
            if self.VARIANT != "v221"
            else None
        )
        projector_state = (
            self.projector.state_dict()
            if self.projector is not None
            else None
        )
        server_optimizer = (
            self.optimizer.state_dict() if self.optimizer else None
        )

        for client_idx in sampled_clients:
            self._set_client_data(client_idx)
            self.client.download_v22(
                body_state,
                linear_state,
                projector_state,
                self.prototypes,
                self.proto_reliability,
                (
                    self.hist_body.state_dict()
                    if self.history_enabled
                    else None
                ),
                (
                    self.hist_linear_head.state_dict()
                    if self.history_enabled and self.VARIANT != "v221"
                    else None
                ),
                (
                    self.hist_projector.state_dict()
                    if self.history_enabled
                    and self.hist_projector is not None
                    else None
                ),
                (
                    self.hist_prototypes if self.history_enabled else None
                ),
                (
                    self.hist_reliability if self.history_enabled else None
                ),
                server_optimizer,
            )
            (
                local_results,
                local_size,
                local_prototypes,
                counts,
                variances,
                diagnostics,
            ) = self.client.train(
                round_number=round_number,
                evaluate_metrics=evaluate_metrics,
                proto_lambda=proto_lambda,
            )

            uploads.append(self.client.upload_local())
            client_sizes.append(local_size)
            prototype_list.append(local_prototypes)
            count_list.append(counts)
            variance_list.append(variances)
            diagnostic_list.append(diagnostics)
            round_results = self._results_updater(
                round_results, local_results
            )
            self.client.reset()

        gate_diagnostics = self._aggregate_diagnostics(
            diagnostic_list, client_sizes
        )
        return (
            uploads,
            client_sizes,
            prototype_list,
            count_list,
            variance_list,
            round_results,
            gate_diagnostics,
        )

    def _aggregate_models(self, uploads, client_sizes):
        total_size = sum(client_sizes)
        components = ["body"]
        if self.VARIANT != "v221":
            components.append("linear_head")
        if self.projector is not None:
            components.append("projector")

        for component in components:
            average = {}
            for key in uploads[0][component]:
                average[key] = sum(
                    upload[component][key]
                    * (client_sizes[index] / total_size)
                    for index, upload in enumerate(uploads)
                )
            getattr(self, component).load_state_dict(average)

    def _aggregate_prototypes(
        self, prototype_list, count_list, variance_list
    ):
        weighted_sum = torch.zeros_like(self.prototypes)
        current_effective_count = torch.zeros(
            self.num_classes, device=self.device
        )
        dispersion_weighted_sum = torch.zeros(
            self.num_classes, device=self.device
        )

        for prototypes, counts, variances in zip(
            prototype_list, count_list, variance_list
        ):
            prototypes = prototypes.to(self.device)
            counts = counts.to(self.device)
            variances = variances.to(self.device)
            reliability = (
                counts / (counts + self.tau_n)
            ) * torch.exp(-variances / self.tau_v)
            reliability = torch.where(
                counts > 0, reliability, torch.zeros_like(reliability)
            )
            effective = reliability * counts
            weighted_sum += prototypes * effective.unsqueeze(1)
            current_effective_count += effective
            dispersion_weighted_sum += variances * effective

        old_prototypes = self.prototypes.clone()
        old_age = self.prototype_age.clone()
        updated_prototypes = old_prototypes.clone()
        updated_count = self.effective_count.clone()
        updated_age = old_age + 1
        updated_reliability = torch.zeros_like(self.proto_reliability)
        updated_alpha = torch.zeros_like(self.alpha_update)

        valid = current_effective_count > 0
        if valid.any():
            decayed_old_count = (
                self.rho ** old_age[valid].float()
            ) * self.effective_count[valid]
            current_proto = F.normalize(
                weighted_sum[valid]
                / (current_effective_count[valid].unsqueeze(1) + self.eps),
                dim=1,
            )
            raw_alpha = current_effective_count[valid] / (
                decayed_old_count
                + current_effective_count[valid]
                + self.eps
            )
            alpha = raw_alpha.clamp(self.alpha_min, self.alpha_max)
            updated_prototypes[valid] = F.normalize(
                (1 - alpha.unsqueeze(1)) * old_prototypes[valid]
                + alpha.unsqueeze(1) * current_proto,
                dim=1,
            )
            updated_count[valid] = (
                decayed_old_count + current_effective_count[valid]
            )
            updated_reliability[valid] = (
                current_effective_count[valid]
                / (
                    current_effective_count[valid]
                    + decayed_old_count
                    + self.tau_R
                )
            ) * torch.exp(-self.age_gamma * old_age[valid].float())
            updated_alpha[valid] = alpha
            updated_age[valid] = 0

        self.prototypes = updated_prototypes
        self.effective_count = updated_count
        self.prototype_age = updated_age
        self.proto_reliability = updated_reliability
        self.alpha_update = updated_alpha
        self._campaign_round_dispersion = torch.where(
            current_effective_count > 0,
            dispersion_weighted_sum
            / current_effective_count.clamp_min(self.eps),
            torch.full_like(current_effective_count, float("nan")),
        )
        self._campaign_proto_drift = F.cosine_similarity(
            updated_prototypes, old_prototypes, dim=1
        )

    def _aggregate_diagnostics(self, diagnostics, client_sizes):
        if not diagnostics:
            return {
                "task_gate": 0.0,
                "sample_gate": 0.0,
                "class_gate": 0.0,
                "eta": 0.0,
            }
        total_size = sum(client_sizes)
        result = {}
        for key in diagnostics[0]:
            if key == "loss_observations":
                result[key] = int(
                    sum(int(item.get(key, 0)) for item in diagnostics)
                )
            elif key.startswith("loss_"):
                observations = sum(
                    int(item.get("loss_observations", 0))
                    for item in diagnostics
                )
                result[key] = (
                    sum(
                        float(item[key])
                        * int(item.get("loss_observations", 0))
                        for item in diagnostics
                    )
                    / observations
                    if observations
                    else float("nan")
                )
            else:
                result[key] = sum(
                    item[key] * (client_sizes[index] / total_size)
                    for index, item in enumerate(diagnostics)
                )
        return result

    def _history_source_states(self):
        if not self.history_enabled:
            return None
        return {
            "body": copy.deepcopy(self.body.state_dict()),
            "linear_head": (
                copy.deepcopy(self.linear_head.state_dict())
                if self.VARIANT != "v221"
                else None
            ),
            "projector": (
                copy.deepcopy(self.projector.state_dict())
                if self.projector is not None
                else None
            ),
        }

    def _update_histories(self, previous_states):
        if not self.history_enabled:
            if previous_states is not None:
                raise ValueError(
                    "history-free server received unexpected history state"
                )
            return
        self._ema_model(
            self.hist_body, previous_states["body"], self.history_alpha
        )
        if self.VARIANT != "v221":
            self._ema_model(
                self.hist_linear_head,
                previous_states["linear_head"],
                self.history_alpha,
            )
        if self.hist_projector is not None:
            self._ema_model(
                self.hist_projector,
                previous_states["projector"],
                self.history_alpha,
            )
        self.hist_prototypes = F.normalize(
            (1 - self.history_alpha) * self.prototypes
            + self.history_alpha * self.hist_prototypes,
            dim=1,
        )
        self.hist_reliability = (
            (1 - self.history_alpha) * self.proto_reliability
            + self.history_alpha * self.hist_reliability
        )

    def _ema_model(self, target_model, source_state, alpha):
        with torch.no_grad():
            for name, parameter in target_model.named_parameters():
                parameter.data = (
                    alpha * source_state[name].to(parameter.device)
                    + (1 - alpha) * parameter.data
                )

    def _evaluate_global(self, round_number=None):
        self.body.eval()
        if self.VARIANT != "v221":
            self.linear_head.eval()
        if self.projector is not None:
            self.projector.eval()

        correct = {
            "main": 0,
            "linear": 0,
            "proto": 0,
            "fused": 0,
        }
        total = 0
        t2_accumulator = (
            T2EvaluationAccumulator(self.num_classes)
            if getattr(self, "t2_diagnostics_enabled", False)
            else None
        )
        campaign_accumulator = (
            ProtoGeometryAccumulator(
                self.num_classes, int(self.prototypes.shape[1])
            )
            if getattr(self, "campaign_diagnostics_enabled", False)
            and self.VARIANT != "v221"
            else None
        )
        sample_prediction_accumulator = (
            SamplePredictionAccumulator()
            if getattr(self, "sample_prediction_logging_enabled", False)
            and (
                round_number is None
                or int(round_number) >= self.sample_prediction_start_round
            )
            else None
        )
        with torch.no_grad():
            for data, targets in self.testloader:
                data, targets = data.to(self.device), targets.to(self.device)
                if self.VARIANT == "v221":
                    features = F.normalize(self.body(data), dim=1)
                    proto_logits = self.scaling * (
                        features @ F.normalize(self.prototypes, dim=1).T
                    )
                    predictions = proto_logits.argmax(dim=1)
                    correct["proto"] += (predictions == targets).sum().item()
                    correct["main"] += (predictions == targets).sum().item()
                else:
                    forward_args = (
                        data,
                        self.body,
                        self.linear_head,
                        self.projector,
                        self.prototypes,
                        self.proto_reliability,
                    )
                    if self.VARIANT == "v224" and hasattr(
                        self.client, "student_evaluation_fusion_mode"
                    ):
                        output = self.client._hybrid_forward(
                            *forward_args,
                            fusion_mode=(
                                self.client.student_evaluation_fusion_mode
                            ),
                        )
                    else:
                        # Non-v224 subclasses sharing this server retain their
                        # established forward signature and semantics.
                        output = self.client._hybrid_forward(*forward_args)
                    linear_predictions = output["linear_logits"].argmax(dim=1)
                    proto_predictions = output["proto_logits"].argmax(dim=1)
                    fused_predictions = output["fused_logits"].argmax(dim=1)
                    # Older shared-server variants (notably ProtoV4.0) do not
                    # expose the campaign-only alias.  Their established main
                    # branch remains the fused branch.
                    main_predictions = output.get(
                        "main_logits", output["fused_logits"]
                    ).argmax(dim=1)
                    correct["linear"] += (
                        linear_predictions == targets
                    ).sum().item()
                    correct["proto"] += (
                        proto_predictions == targets
                    ).sum().item()
                    correct["fused"] += (
                        fused_predictions == targets
                    ).sum().item()
                    if t2_accumulator is not None:
                        t2_accumulator.update(
                            targets,
                            linear_predictions,
                            proto_predictions,
                            fused_predictions,
                            output["sample_gate"],
                            self.proto_reliability,
                        )
                    if campaign_accumulator is not None:
                        campaign_accumulator.update(
                            targets,
                            output["proto_features"],
                            output["proto_cosine_logits"],
                        )
                    if sample_prediction_accumulator is not None:
                        sample_prediction_accumulator.update(
                            targets,
                            linear_predictions,
                            proto_predictions,
                            fused_predictions,
                            main_predictions,
                        )
                    correct["main"] += (
                        main_predictions == targets
                    ).sum().item()
                total += targets.size(0)

        result = {
            key: value / total if total else 0.0
            for key, value in correct.items()
        }
        result.update(
            {
                "main_correct": int(correct["main"]),
                "linear_correct": int(correct["linear"]),
                "proto_correct": int(correct["proto"]),
                "fused_correct": int(correct["fused"]),
                "total": int(total),
            }
        )
        if t2_accumulator is not None:
            result["_t2_diagnostics"] = t2_accumulator.finalize()
        if campaign_accumulator is not None:
            result["_campaign_geometry"] = campaign_accumulator.finalize(
                self.prototypes,
                getattr(self.linear_head, "weight", None),
            )
        if sample_prediction_accumulator is not None:
            result["_sample_predictions"] = (
                sample_prediction_accumulator.finalize()
            )
        return result

    def _prepare_t2_diagnostics(self, completed_round):
        if self.t2_diagnostics is None:
            return
        self.t2_diagnostics.prepare(
            completed_round,
            resuming=self.resume_from is not None,
        )

    def _prepare_campaign_diagnostics(self, completed_round):
        if self.campaign_diagnostics is None:
            return
        self.campaign_diagnostics.prepare(
            completed_round,
            resuming=self.resume_from is not None,
        )

    def _prepare_sample_prediction_diagnostics(self, completed_round):
        if self.sample_prediction_writer is None:
            return
        self.sample_prediction_writer.prepare(
            completed_round,
            resuming=self.resume_from is not None,
        )

    def _commit_sample_prediction_diagnostics(
        self, round_number, sample_predictions
    ):
        if self.sample_prediction_writer is None or sample_predictions is None:
            return
        self.sample_prediction_writer.write_evaluation(
            round_number,
            getattr(self.client, "proto_arm", "full"),
            sample_predictions,
        )

    def _commit_campaign_diagnostics(
        self,
        round_number,
        sampled_clients,
        geometry,
        losses,
    ):
        if self.campaign_diagnostics is None:
            return
        if int(round_number) == 0:
            raw_support = None
            dispersion = torch.full_like(
                self.effective_count, float("nan")
            )
            drift = torch.full_like(
                self.effective_count, float("nan")
            )
        else:
            data_map = np.asarray(self.data_distributed["data_map"])
            selected = np.asarray(sampled_clients, dtype=np.int64)
            raw_support = data_map[selected].sum(axis=0)
            dispersion = self._campaign_round_dispersion
            drift = self._campaign_proto_drift
        self.campaign_diagnostics.write_state(
            round_number,
            raw_support,
            self.effective_count.detach().cpu().numpy(),
            self.prototype_age.detach().cpu().numpy(),
            self.proto_reliability.detach().cpu().numpy(),
            dispersion.detach().cpu().numpy(),
            drift.detach().cpu().numpy(),
        )
        if geometry is not None:
            self.campaign_diagnostics.write_geometry(
                round_number, geometry
            )
        self.campaign_diagnostics.write_losses(
            round_number,
            getattr(self.client, "proto_arm", "full"),
            losses,
        )

    def _commit_t2_diagnostics(
        self, round_number, sampled_clients, evaluation_diagnostics
    ):
        if self.t2_diagnostics is None:
            return
        if int(round_number) == 0:
            support_counts = None
        else:
            data_map = np.asarray(self.data_distributed["data_map"])
            selected = np.asarray(sampled_clients, dtype=np.int64)
            # T2-B support is the raw local-training class count of the
            # selected clients in this round.  It is not prototype count,
            # historical count, aggregation weight, or test-set count.
            support_counts = data_map[selected].sum(axis=0)
        self.t2_diagnostics.write_support(round_number, support_counts)
        if evaluation_diagnostics is not None:
            self.t2_diagnostics.write_evaluation(
                round_number, evaluation_diagnostics
            )

    def _set_client_data(self, client_idx):
        super()._set_client_data(client_idx)
        self.client.major_labels = self.client_y_lst[client_idx]

    def _proto_lambda(self, round_number):
        return proto_lambda_at_round(
            round_number,
            self.proto_lambda_base,
            self.proto_lambda_scale,
            self.proto_loss_start_round,
            self.proto_loss_ramp_rounds,
        )

    def _should_evaluate(self, round_number):
        return super()._should_evaluate(round_number)

    def _current_learning_rate(self):
        return (
            self.optimizer.param_groups[0]["lr"]
            if self.optimizer is not None
            else 0.0
        )

    def _memory_log_values(self):
        values = {}
        counts = self.effective_count.detach().cpu().tolist()
        ages = self.prototype_age.detach().cpu().tolist()
        alphas = self.alpha_update.detach().cpu().tolist()
        reliabilities = self.proto_reliability.detach().cpu().tolist()
        for class_idx in range(self.num_classes):
            prefix = f"proto_memory/class_{class_idx}"
            values[f"{prefix}/effective_count"] = float(counts[class_idx])
            values[f"{prefix}/age"] = int(ages[class_idx])
            values[f"{prefix}/alpha_update"] = float(alphas[class_idx])
            values[f"{prefix}/reliability"] = float(
                reliabilities[class_idx]
            )
        return values

    def export_model_state_dict(self):
        """Export the trained decoupled body/head as the original model.

        ``self.model`` is only a construction/checkpoint compatibility shell
        for this server family; training updates ``self.body`` and
        ``self.linear_head``.  This hook fixes only the terminal ``model.pth``
        export and does not alter checkpoints or runtime state.
        """
        if self.VARIANT != "v224":
            return super().export_model_state_dict()
        exported = copy.deepcopy(self.model).cpu()
        for name in ("conv2d_1", "conv2d_2", "linear_1"):
            getattr(exported, name).load_state_dict(
                getattr(self.body, name).state_dict()
            )
        head_name = "linear_2" if hasattr(exported, "linear_2") else "classifier"
        getattr(exported, head_name).load_state_dict(
            self.linear_head.state_dict()
        )
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in exported.state_dict().items()
        }

    def _print_memory(self):
        def values(tensor, digits=4):
            return [
                round(float(value), digits)
                for value in tensor.detach().cpu().tolist()
            ]

        print(
            "[Proto Memory] "
            f"effective_count={values(self.effective_count)}, "
            f"age={self.prototype_age.detach().cpu().tolist()}, "
            f"alpha_update={values(self.alpha_update)}, "
            f"reliability={values(self.proto_reliability)}"
        )

    def _print_branch_accuracy(self, evaluation):
        print(
            "[Server Branch Acc] "
            f"Linear - {evaluation['linear']:.4f}, "
            f"Proto - {evaluation['proto']:.4f}, "
            f"Fused - {evaluation['fused']:.4f}"
        )

    def _finish_round(
        self,
        round_results,
        gate_diagnostics,
        round_number,
        start_time,
        evaluate_metrics,
        learning_rate,
        proto_lambda,
    ):
        print(
            f"[Round {round_number}/{self.n_rounds}] "
            f"Elapsed {time.perf_counter() - start_time:.1f}s"
        )
        self._print_memory()
        if self.VARIANT != "v221":
            print(
                "[Gate Stat] "
                f"task={gate_diagnostics['task_gate']:.4f}, "
                f"sample_avg={gate_diagnostics['sample_gate']:.4f}, "
                f"class_avg={gate_diagnostics['class_gate']:.4f}, "
                f"eta_avg={gate_diagnostics['eta']:.4f}"
            )

        log_values = {
            "round": round_number,
            "learning_rate": learning_rate,
            "proto_lambda": proto_lambda,
            **self._memory_log_values(),
        }
        if self.VARIANT != "v221":
            log_values.update(
                {
                    f"gates/{key}": value
                    for key, value in gate_diagnostics.items()
                }
            )

        evaluation = None
        eval_sec = 0.0
        if evaluate_metrics:
            eval_start = time.perf_counter()
            evaluation = self._evaluate_global(round_number=round_number)
            eval_sec = time.perf_counter() - eval_start
            server_acc = evaluation["main"]
            self._store_evaluation(round_number, evaluation)
            if self.local_metrics_enabled:
                train_accs = round_results.get("train_acc", [])
                test_accs = round_results.get("test_acc", [])
                print(
                    f"[Local Stat (Train Acc)]: "
                    f"{[round(value, 4) for value in train_accs]}, "
                    f"Avg - {np.mean(train_accs):.4f} "
                    f"(std {np.std(train_accs):.4f})"
                )
                print(
                    f"[Local Stat (Test Acc)]: "
                    f"{[round(value, 4) for value in test_accs]}, "
                    f"Avg - {np.mean(test_accs):.4f} "
                    f"(std {np.std(test_accs):.4f})"
                )
            print(
                f"[Server Stat] Acc - {server_acc:.4f} "
                f"({evaluation['main_correct']}/{evaluation['total']})"
            )
            if self.VARIANT != "v221":
                self._print_branch_accuracy(evaluation)
            log_values["server_test_acc"] = server_acc
            if self.local_metrics_enabled:
                log_values.update(
                    {
                        "local_train_acc": float(np.mean(train_accs)),
                        "local_test_acc": float(np.mean(test_accs)),
                    }
                )
            if self.VARIANT != "v221":
                log_values.update(
                    {
                        "server_linear_acc": evaluation["linear"],
                        "server_proto_acc": evaluation["proto"],
                        "server_fused_acc": evaluation["fused"],
                    }
                )
        else:
            print(
                "[Evaluation] Skipped "
                f"(eval_interval={self.eval_interval})"
            )

        wandb.log(log_values, step=round_number)
        print("-" * 50)
        return evaluation, eval_sec

    def _extra_checkpoint_state(self):
        state = {
            "body": self.body.state_dict(),
            "linear_head": self.linear_head.state_dict(),
            "projector": (
                self.projector.state_dict()
                if self.projector is not None
                else None
            ),
            "scaling": self.scaling.detach().cpu(),
            "prototypes": self.prototypes.detach().cpu(),
            "effective_count": self.effective_count.detach().cpu(),
            "prototype_age": self.prototype_age.detach().cpu(),
            "proto_reliability": self.proto_reliability.detach().cpu(),
            "alpha_update": self.alpha_update.detach().cpu(),
        }
        if self.history_enabled:
            state.update(
                {
                    "hist_body": self.hist_body.state_dict(),
                    "hist_linear_head": self.hist_linear_head.state_dict(),
                    "hist_projector": (
                        self.hist_projector.state_dict()
                        if self.hist_projector is not None
                        else None
                    ),
                    "hist_prototypes": self.hist_prototypes.detach().cpu(),
                    "hist_reliability": self.hist_reliability.detach().cpu(),
                }
            )
        return state

    def _load_extra_checkpoint_state(self, state):
        self.body.load_state_dict(state["body"])
        self.linear_head.load_state_dict(state["linear_head"])
        if self.projector is not None and state["projector"] is not None:
            self.projector.load_state_dict(state["projector"])
        with torch.no_grad():
            self.scaling.copy_(state["scaling"].to(self.device))
        self.prototypes = state["prototypes"].to(self.device)
        self.effective_count = state["effective_count"].to(self.device)
        self.prototype_age = state["prototype_age"].to(self.device)
        self.proto_reliability = state["proto_reliability"].to(self.device)
        self.alpha_update = state["alpha_update"].to(self.device)
        history_keys = {
            "hist_body",
            "hist_linear_head",
            "hist_projector",
            "hist_prototypes",
            "hist_reliability",
        }
        if self.history_enabled:
            missing = history_keys.difference(state)
            if missing:
                raise ValueError(
                    "history-enabled checkpoint is missing: "
                    + ", ".join(sorted(missing))
                )
            self.hist_body.load_state_dict(state["hist_body"])
            self.hist_linear_head.load_state_dict(state["hist_linear_head"])
            if (
                self.hist_projector is not None
                and state["hist_projector"] is not None
            ):
                self.hist_projector.load_state_dict(state["hist_projector"])
            self.hist_prototypes = state["hist_prototypes"].to(self.device)
            self.hist_reliability = state["hist_reliability"].to(self.device)
        elif history_keys.intersection(state):
            raise ValueError(
                "history-free checkpoint unexpectedly contains history state"
            )
