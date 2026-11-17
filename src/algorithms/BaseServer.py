"""Common server runtime for reproducible federated-learning experiments."""

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import wandb

from .measures import evaluate_model_counts, get_round_personalized_acc

__all__ = ["BaseServer"]


def _config_value(config, key, default):
    try:
        return getattr(config, key)
    except (AttributeError, KeyError):
        return default


class BaseServer:
    METRIC_FIELDS = [
        "algorithm",
        "dataset",
        "partition",
        "seed",
        "round",
        "evaluated",
        "correct",
        "total",
        "acc_raw",
        "acc_display",
        "server_linear_acc",
        "server_proto_acc",
        "server_fused_acc",
        "learning_rate",
        "sampled_clients",
        "client_schedule_hash",
        "data_init_sec",
        "train_sec",
        "aggregate_sec",
        "eval_sec",
        "checkpoint_sec",
        "round_total_sec",
        "peak_memory_mb",
        "proto_lambda",
        "anomaly",
    ]

    def __init__(
        self,
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler=None,
        n_rounds=200,
        sample_ratio=0.1,
        local_epochs=5,
        device="cuda:0",
        output_dir=None,
        checkpoint_interval=50,
        save_model_artifacts=True,
        resume_from=None,
        client_schedule_path=None,
        client_schedule_seed=19940817,
        overwrite_output=False,
        run_metadata=None,
    ):
        self.algo_params = algo_params
        self.local_metrics_enabled = bool(
            _config_value(algo_params, "local_metrics_enabled", True)
        )
        legacy_interval = int(_config_value(algo_params, "eval_interval", 1))
        self.server_eval_interval = int(
            _config_value(
                algo_params, "server_eval_interval", legacy_interval
            )
        )
        self.server_eval_dense_start_round = _config_value(
            algo_params, "server_eval_dense_start_round", None
        )
        if self.server_eval_interval <= 0:
            raise ValueError("server_eval_interval must be > 0")
        if self.server_eval_dense_start_round is not None:
            self.server_eval_dense_start_round = int(
                self.server_eval_dense_start_round
            )
            if self.server_eval_dense_start_round <= 0:
                raise ValueError(
                    "server_eval_dense_start_round must be positive or null"
                )

        self.num_classes = data_distributed["num_classes"]
        self.model = model
        self.testloader = data_distributed["global"]["test"]
        self.criterion = nn.CrossEntropyLoss()
        self.data_distributed = data_distributed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_ratio = sample_ratio
        self.n_rounds = n_rounds
        self.device = device
        self.n_clients = len(data_distributed["local"].keys())
        self.local_epochs = local_epochs
        self.checkpoint_interval = int(checkpoint_interval)
        # Storage policy only.  When disabled, round_metrics.csv, train.log,
        # config.json and meta.json remain durable, while all model-bearing
        # checkpoints are skipped.  The default preserves legacy behavior.
        self.save_model_artifacts = bool(save_model_artifacts)
        self.resume_from = Path(resume_from) if resume_from else None
        self.overwrite_output = bool(overwrite_output)
        self.run_metadata = dict(run_metadata or {})
        self.output_dir = Path(output_dir) if output_dir else None
        self.metrics_path = (
            self.output_dir / "round_metrics.csv"
            if self.output_dir is not None
            else None
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.server_results = {
            "client_history": [],
            "test_rounds": [],
            "test_accuracy": [],
            "test_correct": [],
            "test_total": [],
        }

        self.client_schedule_seed = int(client_schedule_seed)
        self.client_schedule_path = (
            Path(client_schedule_path)
            if client_schedule_path
            else (
                self.output_dir / "client_schedule.json"
                if self.output_dir is not None
                else None
            )
        )
        self.client_schedule = self._load_or_create_client_schedule()
        schedule_bytes = json.dumps(
            self.client_schedule,
            separators=(",", ":"),
        ).encode("utf-8")
        self.client_schedule_hash = hashlib.sha256(schedule_bytes).hexdigest()
        self.run_metadata["client_schedule_hash"] = self.client_schedule_hash
        self._write_meta()

    def run(self):
        """Run FL with an auditable evaluation and checkpoint schedule."""
        self._print_start()
        completed_round = self._maybe_resume()

        if completed_round == 0:
            eval_start = time.perf_counter()
            initial = self._evaluate_server()
            eval_sec = time.perf_counter() - eval_start
            self._store_evaluation(0, initial)
            print(
                f"[Round 0/{self.n_rounds}] [Server Stat] "
                f"Acc - {initial['main']:.4f} "
                f"({initial['main_correct']}/{initial['total']})"
            )
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
            )
            print("-" * 50)

        for round_idx in range(completed_round, self.n_rounds):
            round_number = round_idx + 1
            round_start = time.perf_counter()
            learning_rate = self._current_learning_rate()
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            train_start = time.perf_counter()
            updated_local_weights, client_sizes, round_results = (
                self._clients_training(sampled_clients)
            )
            train_sec = time.perf_counter() - train_start

            aggregate_start = time.perf_counter()
            ag_weights = self._aggregation(
                updated_local_weights, client_sizes
            )
            self.model.load_state_dict(ag_weights)
            aggregate_sec = time.perf_counter() - aggregate_start

            evaluation = None
            eval_sec = 0.0
            evaluated = self._should_evaluate(round_number)
            if evaluated:
                eval_start = time.perf_counter()
                evaluation = self._evaluate_server()
                eval_sec = time.perf_counter() - eval_start
                self._store_evaluation(round_number, evaluation)

            if self.local_metrics_enabled:
                eval_results = get_round_personalized_acc(
                    round_results,
                    self.server_results,
                    self.data_distributed,
                )
                wandb.log(eval_results, step=round_number)

            self._step_scheduler()

            self._wandb_logging(
                round_results,
                round_number,
                evaluation=evaluation,
            )
            self._print_stats(
                round_results,
                evaluation,
                round_number,
                time.perf_counter() - round_start,
            )
            # Commit the round record before its checkpoint.  If the process
            # stops between these two writes, resume truncates this row back to
            # the older checkpoint; a checkpoint can therefore never point to
            # an unrecorded training round.
            pre_checkpoint_total_sec = time.perf_counter() - round_start
            self._append_metric_row(
                round_number=round_number,
                evaluated=evaluated,
                evaluation=evaluation,
                learning_rate=learning_rate,
                sampled_clients=sampled_clients,
                train_sec=train_sec,
                aggregate_sec=aggregate_sec,
                eval_sec=eval_sec,
                checkpoint_sec=0.0,
                round_total_sec=pre_checkpoint_total_sec,
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
                            f"{time.perf_counter() - round_start:.9f}"
                        ),
                    },
                )
            print("-" * 50)

    def _clients_training(self, sampled_clients):
        updated_local_weights, client_sizes = [], []
        round_results = {}
        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        for client_idx in sampled_clients:
            self._set_client_data(client_idx)
            self.client.download_global(server_weights, server_optimizer)
            local_results, local_size = self.client.train()
            updated_local_weights.append(self.client.upload_local())
            round_results = self._results_updater(
                round_results, local_results
            )
            client_sizes.append(local_size)
            self.client.reset()

        return updated_local_weights, client_sizes, round_results

    def _load_or_create_client_schedule(self):
        clients_per_round = max(int(self.n_clients * self.sample_ratio), 1)
        if self.client_schedule_path is not None and self.client_schedule_path.exists():
            with self.client_schedule_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            schedule = payload.get("schedule", payload)
        else:
            rng = np.random.RandomState(self.client_schedule_seed)
            schedule = [
                rng.choice(
                    self.n_clients,
                    clients_per_round,
                    replace=False,
                ).astype(int).tolist()
                for _ in range(self.n_rounds)
            ]
            if self.client_schedule_path is not None:
                self.client_schedule_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                payload = {
                    "seed": self.client_schedule_seed,
                    "n_clients": self.n_clients,
                    "clients_per_round": clients_per_round,
                    "n_rounds": self.n_rounds,
                    "schedule": schedule,
                }
                self._atomic_json_dump(payload, self.client_schedule_path)

        if len(schedule) != self.n_rounds:
            raise ValueError(
                "client schedule length does not match n_rounds: "
                f"{len(schedule)} != {self.n_rounds}"
            )
        for round_number, clients in enumerate(schedule, start=1):
            if len(clients) != clients_per_round:
                raise ValueError(
                    f"round {round_number} has an invalid client count"
                )
            if len(set(clients)) != len(clients):
                raise ValueError(
                    f"round {round_number} contains duplicate clients"
                )
            if any(client < 0 or client >= self.n_clients for client in clients):
                raise ValueError(
                    f"round {round_number} contains an out-of-range client"
                )
        return [[int(client) for client in clients] for clients in schedule]

    def _client_sampling(self, round_idx):
        return np.asarray(self.client_schedule[round_idx], dtype=int)

    def _should_evaluate(self, round_number):
        return (
            round_number % self.server_eval_interval == 0
            or (
                self.server_eval_dense_start_round is not None
                and round_number >= self.server_eval_dense_start_round
            )
            or round_number == self.n_rounds
        )

    def _evaluate_server(self):
        result = evaluate_model_counts(
            self.model,
            self.testloader,
            device=self.device,
        )
        return {
            "main": result["accuracy"],
            "main_correct": result["correct"],
            "total": result["total"],
        }

    def _store_evaluation(self, round_number, evaluation):
        self.server_results["test_rounds"].append(int(round_number))
        self.server_results["test_accuracy"].append(
            float(evaluation["main"])
        )
        self.server_results["test_correct"].append(
            int(evaluation["main_correct"])
        )
        self.server_results["test_total"].append(int(evaluation["total"]))

    def _set_client_data(self, client_idx):
        self.client.datasize = self.data_distributed["local"][client_idx][
            "datasize"
        ]
        self.client.trainloader = self.data_distributed["local"][client_idx][
            "train"
        ]
        self.client.testloader = (
            self.data_distributed["global"]["test"]
            if self.local_metrics_enabled
            else None
        )

    def _aggregation(self, weights, client_sizes):
        proportions = torch.tensor(client_sizes, dtype=torch.float)
        proportions /= torch.sum(proportions)
        average = copy.deepcopy(weights[0])
        for key in average:
            average[key] = average[key] * proportions[0]
        for key in average:
            for index in range(1, len(weights)):
                average[key] += weights[index][key] * proportions[index]
        return copy.deepcopy(average)

    def _results_updater(self, round_results, local_results):
        for key, item in local_results.items():
            round_results.setdefault(key, []).append(item)
        return round_results

    def _current_learning_rate(self):
        if self.optimizer is None:
            return 0.0
        return float(self.optimizer.param_groups[0]["lr"])

    def _step_scheduler(self):
        """Advance the server LR controller without changing model weights.

        Local client optimizers perform the real SGD steps.  The server
        optimizer only carries the LR and optimizer template distributed to
        clients, so PyTorch cannot observe those local steps and otherwise
        emits a false scheduler-order warning.  A no-gradient optimizer step
        updates only PyTorch's bookkeeping before advancing the unchanged LR
        schedule.
        """
        if self.scheduler is None:
            return
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.step()
        self.scheduler.step()

    def _print_start(self):
        status = (
            "enabled"
            if self.local_metrics_enabled
            else "disabled (no Local Train/Test Acc forward passes)"
        )
        print(f"Local metrics: {status}")
        print(
            "Server evaluation: R0, every "
            f"{self.server_eval_interval} rounds"
            + (
                f", and every round from R{self.server_eval_dense_start_round}"
                if self.server_eval_dense_start_round is not None
                else ""
            )
        )
        print(
            "Client schedule SHA256: "
            f"{self.client_schedule_hash}"
        )
        if self.device == "cpu":
            return "cpu"
        device_name = None
        if self.device == "mps":
            device_name = "Apple MPS (Metal Performance Shaders)"
        elif isinstance(self.device, str) and self.device.startswith("cuda"):
            device_idx = (
                int(self.device.split(":")[-1]) if ":" in self.device else 0
            )
            device_name = torch.cuda.get_device_name(device_idx)
        elif isinstance(self.device, torch.device):
            if self.device.type == "cuda":
                device_name = torch.cuda.get_device_name(self.device.index)
            elif self.device.type == "mps":
                device_name = "Apple MPS"
        print("")
        print("=" * 50)
        print(f"Train start on device: {device_name or self.device}")
        print("=" * 50)

    def _print_stats(
        self,
        round_results,
        evaluation,
        round_number,
        round_elapse,
    ):
        print(
            f"[Round {round_number}/{self.n_rounds}] "
            f"Elapsed {round_elapse:.1f}s "
            f"(Current Time: {time.strftime('%H:%M:%S')})"
        )
        if self.local_metrics_enabled:
            print(
                "[Local Stat (Train Acc)]: {}, Avg - {:.4f} "
                "(std {:.4f})".format(
                    round_results["train_acc"],
                    np.mean(round_results["train_acc"]),
                    np.std(round_results["train_acc"]),
                )
            )
            print(
                "[Local Stat (Test Acc)]: {}, Avg - {:.4f} "
                "(std {:.4f})".format(
                    round_results["test_acc"],
                    np.mean(round_results["test_acc"]),
                    np.std(round_results["test_acc"]),
                )
            )
        if evaluation is None:
            print("[Server Stat] Evaluation skipped")
        else:
            print(
                f"[Server Stat] Acc - {evaluation['main']:.4f} "
                f"({evaluation['main_correct']}/{evaluation['total']})"
            )

    def _wandb_logging(
        self,
        round_results,
        round_number,
        evaluation=None,
    ):
        values = {"learning_rate": self._current_learning_rate()}
        if self.local_metrics_enabled:
            values.update(
                {
                    "local_train_acc": np.mean(round_results["train_acc"]),
                    "local_test_acc": np.mean(round_results["test_acc"]),
                }
            )
        if evaluation is not None:
            values["server_test_acc"] = evaluation["main"]
        wandb.log(values, step=round_number)

    def _append_metric_row(
        self,
        round_number,
        evaluated,
        evaluation,
        learning_rate,
        sampled_clients,
        train_sec,
        aggregate_sec,
        eval_sec,
        checkpoint_sec,
        round_total_sec,
        extra=None,
    ):
        if self.metrics_path is None:
            return
        evaluation = evaluation or {}
        accuracy = evaluation.get("main")
        anomaly = ""
        if accuracy is not None and (
            not math.isfinite(float(accuracy))
            or not 0.0 <= float(accuracy) <= 1.0
        ):
            anomaly = "invalid_accuracy"
        row = {
            "algorithm": self.run_metadata.get("algorithm", ""),
            "dataset": self.run_metadata.get("dataset", ""),
            "partition": self.run_metadata.get("partition", ""),
            "seed": self.run_metadata.get("seed", ""),
            "round": int(round_number),
            "evaluated": int(bool(evaluated)),
            "correct": evaluation.get("main_correct", ""),
            "total": evaluation.get("total", ""),
            "acc_raw": "" if accuracy is None else repr(float(accuracy)),
            "acc_display": "" if accuracy is None else f"{accuracy:.4f}",
            "server_linear_acc": evaluation.get("linear", ""),
            "server_proto_acc": evaluation.get("proto", ""),
            "server_fused_acc": evaluation.get("fused", ""),
            "learning_rate": repr(float(learning_rate)),
            "sampled_clients": json.dumps(
                [int(client) for client in sampled_clients],
                separators=(",", ":"),
            ),
            "client_schedule_hash": self.client_schedule_hash,
            "data_init_sec": repr(
                float(self.run_metadata.get("data_init_sec", 0.0))
            ),
            "train_sec": f"{train_sec:.9f}",
            "aggregate_sec": f"{aggregate_sec:.9f}",
            "eval_sec": f"{eval_sec:.9f}",
            "checkpoint_sec": f"{checkpoint_sec:.9f}",
            "round_total_sec": f"{round_total_sec:.9f}",
            "peak_memory_mb": f"{self._peak_memory_mb():.3f}",
            "proto_lambda": "",
            "anomaly": anomaly,
        }
        if extra:
            for key, value in extra.items():
                if key in row:
                    row[key] = value
        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    def _checkpoint_if_due(self, round_number):
        if (
            not self.save_model_artifacts
            or self.output_dir is None
            or self.checkpoint_interval <= 0
        ):
            return False
        if (
            round_number % self.checkpoint_interval != 0
            and round_number != self.n_rounds
        ):
            return False
        latest = self.output_dir / "checkpoint_latest.pt"
        self._save_checkpoint(latest, round_number)
        if round_number == self.n_rounds:
            shutil.copy2(latest, self.output_dir / "checkpoint_final.pt")
        return True

    def _save_checkpoint(self, path, round_number):
        state = {
            "format_version": 2,
            "completed_round": int(round_number),
            "metrics_committed_through": int(round_number),
            "client_schedule_hash": self.client_schedule_hash,
            "model": self.model.state_dict(),
            "optimizer": (
                self.optimizer.state_dict()
                if self.optimizer is not None
                else None
            ),
            "scheduler": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "server_results": self.server_results,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "extra_state": self._extra_checkpoint_state(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def _maybe_resume(self):
        if self.resume_from is None:
            self._prepare_metrics_csv(0, resuming=False)
            return 0
        checkpoint = torch.load(self.resume_from, map_location="cpu")
        if checkpoint["client_schedule_hash"] != self.client_schedule_hash:
            raise ValueError("checkpoint client schedule hash mismatch")
        self.model.load_state_dict(checkpoint["model"])
        if self.optimizer is not None and checkpoint["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self._optimizer_to_device()
        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.server_results = checkpoint["server_results"]
        self._load_extra_checkpoint_state(checkpoint.get("extra_state", {}))
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        completed_round = int(checkpoint["completed_round"])
        if (
            checkpoint.get("metrics_committed_through") is not None
            and completed_round > 0
            and (
                self.metrics_path is None
                or not self.metrics_path.exists()
            )
        ):
            raise RuntimeError(
                "auditable checkpoint resume requires its original "
                "round_metrics.csv"
            )
        self._prepare_metrics_csv(completed_round, resuming=True)
        print(
            f"Resumed from {self.resume_from} after round "
            f"{completed_round}."
        )
        return completed_round

    def _extra_checkpoint_state(self):
        return {}

    def export_model_state_dict(self):
        """Return the trained model state used for the final lightweight export.

        Algorithms that train a decoupled representation may override this
        hook.  Checkpoint semantics are intentionally unchanged.
        """
        return self.model.state_dict()

    def _load_extra_checkpoint_state(self, state):
        del state

    def _optimizer_to_device(self):
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _prepare_metrics_csv(self, completed_round, resuming):
        if self.metrics_path is None or not self.metrics_path.exists():
            return
        if not resuming:
            if self.overwrite_output:
                self.metrics_path.unlink()
                return
            raise FileExistsError(
                f"metrics already exist: {self.metrics_path}; set "
                "overwrite_output=true only for disposable runs"
            )
        with self.metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [
                row
                for row in reader
                if int(row["round"]) <= completed_round
            ]
        if completed_round > 0 and not any(
            int(row["round"]) == completed_round for row in rows
        ):
            raise RuntimeError(
                "checkpoint is ahead of round_metrics.csv: missing committed "
                f"round {completed_round}"
            )
        with self.metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())

    def _update_metric_row(self, round_number, updates):
        """Atomically update a committed round row after checkpoint I/O."""
        if self.metrics_path is None or not self.metrics_path.exists():
            return
        with self.metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        matched = False
        for row in rows:
            if int(row["round"]) == int(round_number):
                row.update(updates)
                matched = True
                break
        if not matched:
            raise RuntimeError(
                f"cannot update missing metrics row {round_number}"
            )
        temporary = self.metrics_path.with_suffix(
            self.metrics_path.suffix + ".tmp"
        )
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.metrics_path)

    @staticmethod
    def _peak_memory_mb():
        """Return process peak RSS in MiB (includes unified MPS memory)."""
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return peak / (1024.0 * 1024.0)
        return peak / 1024.0

    def _write_meta(self):
        if self.output_dir is None:
            return
        metadata = {
            **self.run_metadata,
            "n_rounds": self.n_rounds,
            "n_clients": self.n_clients,
            "sample_ratio": self.sample_ratio,
            "local_epochs": self.local_epochs,
            "device": str(self.device),
            "server_eval_interval": self.server_eval_interval,
            "server_eval_dense_start_round": (
                self.server_eval_dense_start_round
            ),
            "checkpoint_interval": self.checkpoint_interval,
            "save_model_artifacts": self.save_model_artifacts,
            "client_schedule_seed": self.client_schedule_seed,
            "client_schedule_hash": self.client_schedule_hash,
            "local_metrics_enabled": self.local_metrics_enabled,
        }
        self._atomic_json_dump(metadata, self.output_dir / "meta.json")

    @staticmethod
    def _atomic_json_dump(payload, path):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    # Retained for external callers and legacy tests. New runs use run().
    def _update_and_evaluate(
        self, ag_weights, round_results, round_idx, start_time
    ):
        self.model.load_state_dict(ag_weights)
        evaluation = self._evaluate_server()
        self._store_evaluation(round_idx + 1, evaluation)
        if self.local_metrics_enabled:
            eval_results = get_round_personalized_acc(
                round_results, self.server_results, self.data_distributed
            )
            wandb.log(eval_results, step=round_idx)
        self._step_scheduler()
        elapsed = time.time() - start_time
        self._wandb_logging(round_results, round_idx, evaluation)
        self._print_stats(round_results, evaluation, round_idx + 1, elapsed)
        print("-" * 50)
