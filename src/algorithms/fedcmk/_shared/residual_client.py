"""Reliability-aware training with No-R inference residuals."""

from algorithms.fedcmk._shared.training_client import (
    ClientTrainer as PrototypeClientTrainer,
)


class ClientTrainer(PrototypeClientTrainer):
    """Use the full training objective and No-R evaluation fusion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.proto_arm != "full":
            raise ValueError("FedCMK requires the Original Full training arm")
        if self.student_training_fusion_mode != "r":
            raise RuntimeError("FedCMK training must remain reliability-aware")
        if not self.teacher_decision_reliability_enabled:
            raise RuntimeError("FedCMK teachers must remain reliability-aware")

        # Final Method B changes one prediction-side edge only:
        # eta_test(i,c) = eta_max * task_gate * uncertainty(i), with the
        # resulting per-sample coefficient broadcast across candidate classes.
        # Training continues to use the complete class reliability vector.
        self.student_evaluation_fusion_mode = "no_r"


__all__ = ["ClientTrainer"]
