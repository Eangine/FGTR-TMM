import torch
import torch.nn.functional as F


def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8):
    batch_size = scores.size(0)
    pid = pid.view(batch_size, 1)
    labels = (pid == pid.t()).float()
    labels_distribute = labels / labels.sum(dim=1, keepdim=True).clamp_min(epsilon)
    logits_t2i = logit_scale * scores
    logits_i2t = logits_t2i.t()
    loss_t2i = -(labels_distribute * F.log_softmax(logits_t2i, dim=1)).sum(dim=1)
    loss_i2t = -(labels_distribute * F.log_softmax(logits_i2t, dim=1)).sum(dim=1)
    return loss_t2i + loss_i2t


def compute_InfoNCE_per(scores, logit_scale):
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()
    targets = torch.arange(scores.size(0), device=scores.device)
    loss_i = F.cross_entropy(logits_per_image, targets, reduction="none")
    loss_t = F.cross_entropy(logits_per_text, targets, reduction="none")
    return 0.5 * (loss_i + loss_t)


def compute_TAL_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))
    labels = (pid == pid.t()).float().to(scores.device)
    mask = 1 - labels
    scaled = (scores / tau).exp()
    alpha_i2t = (scaled * labels / (scaled * labels).sum(dim=1, keepdim=True).clamp_min(1e-8)).detach()
    alpha_t2i = (scaled.t() * labels / (scaled.t() * labels).sum(dim=1, keepdim=True).clamp_min(1e-8)).detach()
    loss_i2t = (- (alpha_i2t * scores).sum(1) + tau * (scaled * mask).sum(1).clamp(max=1e36).log() + margin).clamp(min=0)
    loss_t2i = (- (alpha_t2i * scores.t()).sum(1) + tau * (scaled.t() * mask).sum(1).clamp(max=1e36).log() + margin).clamp(min=0)
    return loss_i2t + loss_t2i


def _resolve_logit_scale(args, logit_scale):
    if logit_scale is not None:
        return logit_scale
    return 1.0 / max(float(args.temperature), 1e-8)


def compute_per_loss(sims, pid, args, logit_scale=None, hist_weights=None, mode=None):
    cur_weights = None
    if mode != "GMM":
        with torch.no_grad():
            scaled_sims = sims / 0.07
            i2t_prob = F.softmax(scaled_sims, dim=1)
            t2i_prob = F.softmax(scaled_sims, dim=0)
            w_tilde = 0.5 * (i2t_prob.diag() + t2i_prob.diag())
            alpha = getattr(args, "momentum_alpha", 0.6)
            cur_weights = alpha * hist_weights + (1.0 - alpha) * w_tilde if hist_weights is not None else w_tilde

    scale = _resolve_logit_scale(args, logit_scale)
    loss_names = args.loss_names
    if "TAL" in loss_names:
        per_loss = compute_TAL_per(sims, pid, 0.015, 0.1)
    elif "InfoNCE" in loss_names:
        per_loss = compute_InfoNCE_per(sims, scale)
    elif "SDM" in loss_names:
        per_loss = compute_sdm_per(sims, pid, scale)
    else:
        raise ValueError(f"Unsupported loss_names: {loss_names}")

    return per_loss, cur_weights


def compute_align_loss(
    sims,
    pid,
    args,
    logit_scale=None,
    hist_weights=None,
    sample_weights=None,
):
    per_loss, cur_weights = compute_per_loss(
        sims,
        pid,
        args,
        logit_scale=logit_scale,
        hist_weights=hist_weights,
    )

    if cur_weights is not None:
        weighted_per_loss = per_loss * cur_weights
    else:
        weighted_per_loss = per_loss

    if sample_weights is not None:
        sample_weights = sample_weights.view(-1).float().to(per_loss.device)
        weighted_per_loss = weighted_per_loss * sample_weights
        valid_sum = sample_weights.sum().clamp_min(1e-8)
        return weighted_per_loss.sum() / valid_sum, cur_weights

    return weighted_per_loss.mean(), cur_weights