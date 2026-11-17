"""FedCMK-c server: no server-side historical model is maintained."""

from algorithms.fedcmk._shared.prototype_server import CountAgeServer
from algorithms.fedcmk.ClientTrainer import ClientTrainer


class Server(CountAgeServer):
    VARIANT = "v224"
    DISPLAY_NAME = "FedCMK-c"
    CLIENT_CLASS = ClientTrainer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # CountAgeServer constructs Original-Full history by default.  Remove
        # it before any training round: FedCMK-c has no server historical
        # model, no server EMA update, and no historical downlink payload.
        self.history_enabled = False
        self.hist_body = None
        self.hist_linear_head = None
        self.hist_projector = None
        self.hist_prototypes = None
        self.hist_reliability = None
        self.run_metadata.update(
            {
                "algorithm_identity": "FedCMK_c",
                "historical_ema_teacher_present": True,
                "historical_teacher_scope": "private_per_client_local_model",
                "server_historical_model_present": False,
                "server_historical_model_downlink": False,
                "client_local_history_initialization": (
                    "first_successful_local_model"
                ),
                "client_local_history_ema": (
                    "H_k=(1-alpha)*H_k+alpha*W_k_local"
                ),
                "server_inference_rule": "fedcmk_no_r_fused",
                "student_training_fusion_mode": "r",
                "student_evaluation_fusion_mode": "no_r",
                "training_reliability_path": "enabled",
            }
        )
        self._assert_fedcmk_c_identity()
        self._write_meta()

    def _assert_fedcmk_c_identity(self):
        """Fail closed if the frozen FedCMK-c identity drifts."""
        checks = {
            "proto_arm": self.client.proto_arm == "full",
            "training_R": self.client.student_training_fusion_mode == "r",
            "evaluation_NoR": (
                self.client.student_evaluation_fusion_mode == "no_r"
            ),
            "teacher_R": self.client.teacher_decision_reliability_enabled,
            "server_history_disabled": not self.history_enabled,
            "server_history_absent": all(
                getattr(self, name) is None
                for name in (
                    "hist_body",
                    "hist_linear_head",
                    "hist_projector",
                    "hist_prototypes",
                    "hist_reliability",
                )
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "FedCMK-c frozen identity violation: " + ", ".join(failed)
            )

    def _evaluate_global(self, round_number=None):
        self._assert_fedcmk_c_identity()
        return super()._evaluate_global(round_number=round_number)

    def _set_client_data(self, client_idx):
        self.client.bind_client(client_idx)
        super()._set_client_data(client_idx)

    def _extra_checkpoint_state(self):
        state = super()._extra_checkpoint_state()
        forbidden = {
            "hist_body",
            "hist_linear_head",
            "hist_projector",
            "hist_prototypes",
            "hist_reliability",
        }
        if forbidden.intersection(state):
            raise RuntimeError("FedCMK-c checkpoint contains server history")
        state["client_local_history"] = (
            self.client.export_local_history_state()
        )
        return state

    def _load_extra_checkpoint_state(self, state):
        state = dict(state)
        client_local_history = state.pop("client_local_history", None)
        if client_local_history is None:
            raise ValueError(
                "FedCMK-c checkpoint is missing client_local_history"
            )
        super()._load_extra_checkpoint_state(state)
        self.client.load_local_history_state(client_local_history)


__all__ = ["Server"]
