"""FedCMK-c client with a private per-client EMA historical teacher.

The trainable student still starts from the current global model, as required
by the federated protocol.  The *historical* teacher is different: it is made
only from this client's own successful local-training results.  It is never
initialized from, replaced by, uploaded to, or shared through a server-side
historical model.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from algorithms.fedcmk._shared.residual_client import ClientTrainer as FedCMKClientTrainer


def _cpu_state_dict(module):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


class ClientTrainer(FedCMKClientTrainer):
    """Maintain one isolated historical teacher for every logical client."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_client_id = None
        self._local_history = {}
        # A first-time client has no legitimate historical teacher.  The flag
        # is switched on only when that client's own stored state is loaded.
        self.history_enabled = False
        self.local_history_alpha = float(self.algo_params.history_alpha)
        if not 0.0 <= self.local_history_alpha <= 1.0:
            raise ValueError("history_alpha must satisfy 0 <= alpha <= 1")

    def bind_client(self, client_id):
        client_id = int(client_id)
        if client_id < 0:
            raise ValueError("client_id must be non-negative")
        self._active_client_id = client_id

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
        if self._active_client_id is None:
            raise RuntimeError("bind_client(client_id) is required before download")
        if any(
            value is not None
            for value in (
                hist_body_state,
                hist_linear_head_state,
                hist_projector_state,
                hist_prototypes,
                hist_reliability,
            )
        ):
            raise ValueError(
                "FedCMK-c forbids server-downlinked historical teacher state"
            )

        local_history = self._local_history.get(self._active_client_id)
        self.history_enabled = local_history is not None

        # The private teacher owns the historical model parameters.  FedCMK's
        # current global prototype/reliability memory remains the shared
        # knowledge basis for both current and historical model forwards; it
        # is not a server-side historical model.
        super().download_v22(
            body_state,
            linear_head_state,
            projector_state,
            prototypes,
            proto_reliability,
            local_history["body"] if local_history is not None else None,
            (
                local_history["linear_head"]
                if local_history is not None
                else None
            ),
            (
                local_history["projector"]
                if local_history is not None
                else None
            ),
            prototypes if local_history is not None else None,
            proto_reliability if local_history is not None else None,
            server_optimizer,
        )

    def train(self, round_number, evaluate_metrics=False, proto_lambda=0.0):
        result = super().train(
            round_number=round_number,
            evaluate_metrics=evaluate_metrics,
            proto_lambda=proto_lambda,
        )
        # Commit the private teacher only after a successful local train.
        self._update_private_history()
        return result

    def _update_private_history(self):
        if self._active_client_id is None:
            raise RuntimeError("cannot update local history without a client_id")
        source = {
            "body": _cpu_state_dict(self.body),
            "linear_head": _cpu_state_dict(self.linear_head),
            "projector": (
                _cpu_state_dict(self.projector)
                if self.projector is not None
                else None
            ),
        }
        previous = self._local_history.get(self._active_client_id)
        if previous is None:
            # First participation: the first historical teacher is the first
            # completed *local* model, never the downloaded global model.
            self._local_history[self._active_client_id] = source
            return

        alpha = self.local_history_alpha
        self._local_history[self._active_client_id] = {
            component: self._ema_state(previous[component], source[component], alpha)
            if source[component] is not None
            else None
            for component in ("body", "linear_head", "projector")
        }

    @staticmethod
    def _ema_state(previous, local, alpha):
        if previous is None or local is None:
            if previous is not None or local is not None:
                raise ValueError("local-history component structure changed")
            return None
        if set(previous) != set(local):
            raise ValueError("local-history state keys changed")
        updated = {}
        for name in previous:
            old_tensor = previous[name]
            local_tensor = local[name]
            if old_tensor.shape != local_tensor.shape:
                raise ValueError(f"local-history tensor shape changed: {name}")
            if torch.is_floating_point(old_tensor):
                # Same update-rate convention as FedCMK's historical model:
                # H_k <- (1-alpha) H_k + alpha W_k^local.
                updated[name] = (
                    (1.0 - alpha) * old_tensor + alpha * local_tensor
                ).clone()
            else:
                updated[name] = local_tensor.clone()
        return updated

    def export_local_history_state(self):
        return {
            int(client_id): {
                component: (
                    {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in state.items()
                    }
                    if state is not None
                    else None
                )
                for component, state in bundle.items()
            }
            for client_id, bundle in self._local_history.items()
        }

    def load_local_history_state(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("client_local_history must be a mapping")
        restored = {}
        for raw_client_id, bundle in state.items():
            client_id = int(raw_client_id)
            if set(bundle) != {"body", "linear_head", "projector"}:
                raise ValueError("invalid client-local history bundle")
            restored[client_id] = {
                component: (
                    {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in component_state.items()
                    }
                    if component_state is not None
                    else None
                )
                for component, component_state in bundle.items()
            }
        self._local_history = restored

    @property
    def local_history_client_ids(self):
        return tuple(sorted(self._local_history))

    def reset(self):
        super().reset()
        self.history_enabled = False
        self._active_client_id = None


__all__ = ["ClientTrainer"]
