import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LMD_Hist_Loss", "proto_lambda_at_round"]


def proto_lambda_at_round(
    round_number,
    proto_lambda_base=1.0,
    proto_lambda_scale=0.25,
    proto_loss_start_round=100,
    proto_loss_ramp_rounds=100,
):
    """Return the auxiliary prototype-loss weight for a 1-based round."""
    if round_number < 1:
        raise ValueError("round_number must be 1-based and >= 1")
    if proto_loss_ramp_rounds <= 0:
        raise ValueError("proto_loss_ramp_rounds must be > 0")
    if round_number <= proto_loss_start_round:
        return 0.0

    progress = min(
        (round_number - proto_loss_start_round) / proto_loss_ramp_rounds,
        1.0,
    )
    return float(proto_lambda_base * proto_lambda_scale * progress)


def lmd_criterion(
    logits_student,
    logits_teacher,
    target,
    T,
    major_labels,
):
    """
    Label-Masking Distillation: mask the ground-truth class and the client's
    major labels from the teacher's soft targets, then distill.
    """
    bs = logits_student.size(0)
    gt_mask = _get_gt_mask(logits_student, target)
    label_mask = torch.zeros_like(logits_student).scatter_(
        1, major_labels.repeat(bs, 1), 1
    ).bool()

    pred_t = F.softmax(logits_teacher / T - 1000 * label_mask - 1000 * gt_mask, dim=1)
    pred_s = F.log_softmax(logits_student / T - 1000 * gt_mask, dim=1)

    return nn.KLDivLoss(reduction="batchmean")(pred_s, pred_t) * (T ** 2)


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


class LMD_Hist_Loss(nn.Module):
    """
    FedLMD-Hist loss: CE + beta * LMD(global) + gamma * LMD(history).

    The "history model" is the global model from the **previous** communication
    round.  Distilling from it adds temporal stability — the local update is
    regularised not only against the current global consensus but also against
    where the consensus was one round ago.
    """

    def __init__(self, num_classes=10, tau=1, beta=1, gamma=0.5):
        super(LMD_Hist_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets, dg_logits, hist_logits, major_labels):
        # supervised loss
        ce_loss = self.CE(logits, targets)

        # current-global distillation (same as original FedLMD)
        lmd_loss = lmd_criterion(logits, dg_logits, targets, self.tau, major_labels)

        # history distillation (same decoupling / label-masking logic)
        hist_loss = lmd_criterion(logits, hist_logits, targets, self.tau, major_labels)

        loss = ce_loss + self.beta * lmd_loss + self.gamma * hist_loss
        return loss
