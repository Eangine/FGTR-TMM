import math
import torch
import torch.nn as nn
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

    loss_i = F.cross_entropy(logits_per_image, targets, reduction='none')
    loss_t = F.cross_entropy(logits_per_text, targets, reduction='none')
    return 0.5 * (loss_i + loss_t)

def compute_TAL_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) 
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().to(scores.device)
    mask = 1 - labels
    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    return loss

def compute_per_loss(sims, pid, args, logit_scale=50, hist_weights=None, mode=None):

    if mode!='GMM':
        scores = sims 
        
        with torch.no_grad():
            scaled_sims = scores/0.07
            # 双向概率估计 (论文公式 7)
            i2t_prob = F.softmax(scaled_sims, dim=1)
            t2i_prob = F.softmax(scaled_sims, dim=0)
            w_tilde = 0.5 * (i2t_prob.diag() + t2i_prob.diag())
            
            alpha = getattr(args, 'momentum_alpha', 0.6) # 动量系数，可在 args 中配置
            if hist_weights is not None:
                cur_weights = alpha * hist_weights + (1.0 - alpha) * w_tilde
            else:
                cur_weights = w_tilde
    else:
        cur_weights = None

    scores = sims 
    logit_scale = 1 / args.temperature

    if 'TAL' in args.loss_names:
        per_loss = compute_TAL_per(scores, pid, 0.015, 0.1)
    elif 'InfoNCE' in args.loss_names:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in args.loss_names:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    else:
        exit()
    
    return per_loss, cur_weights

def compute_align_loss(
    sims,
    pid,
    args,
    label_hat=None,
    logit_scale=50,
    hist_weights=None,
    sample_weights=None
):

    per_loss, cur_weights = compute_per_loss(
        sims, pid, args, logit_scale, hist_weights=hist_weights
    )

    weighted_per_loss = per_loss * cur_weights

    if sample_weights is not None:
        sample_weights = sample_weights.view(-1).float().to(per_loss.device)
        weighted_per_loss = weighted_per_loss * sample_weights
        valid_sum = sample_weights.sum().clamp_min(1e-8)
        final_loss = weighted_per_loss.sum() / valid_sum
        return final_loss, cur_weights

    if label_hat is None:
        return weighted_per_loss.mean(), cur_weights

    label_hat = label_hat.view(-1).float().to(per_loss.device)
    valid_sum = label_hat.sum()

    if valid_sum.item() <= 0:
        return per_loss.new_tensor(0.0), cur_weights

    final_loss = (weighted_per_loss * label_hat).sum() / valid_sum
    return final_loss, cur_weights
