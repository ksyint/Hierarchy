import torch
import torch.nn.functional as F


def harm_dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected,
                  margin, beta=0.1, reduction='mean'):
    """Eq. (5). gamma shifts beta * log-ratio gap (not the unscaled gap)."""
    gap = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    loss = -F.logsigmoid(beta * gap - margin)
    if reduction == 'none':
        return loss
    if reduction != 'mean':
        raise ValueError('reduction must be mean or none')
    return loss.mean()


def ccr_loss(logits, counterfactual_logits, labels, counterfactual_labels):
    """Eq. (7): paired decision NLL on flipped intent, symmetric KL otherwise.

    Decision distributions have class order [comply, refuse]. For same-label
    pairs return 1/2 symmetric-KL, giving lambda/2 scaling in the total loss.
    """
    log_p = logits.log_softmax(-1)
    log_q = counterfactual_logits.log_softmax(-1)
    labels, counterfactual_labels = labels.long(), counterfactual_labels.long()
    paired_nll = F.nll_loss(log_p, labels, reduction='none') + F.nll_loss(log_q, counterfactual_labels, reduction='none')
    symmetric_kl = 0.5 * ((log_p.exp() * (log_p - log_q)).sum(-1) +
                          (log_q.exp() * (log_q - log_p)).sum(-1))
    return torch.where(labels != counterfactual_labels, paired_nll, symmetric_kl).mean()
