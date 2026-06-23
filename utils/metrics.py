
from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import sys


def z_score_norm(matrix):
    mean = matrix.mean()
    std = matrix.std()
    return (matrix - mean) / (std + 1e-8)


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("RASC.eval")

    def _extract_all_features(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        qids, gids = [], []
        text_feats_global = []
        img_feats_global = []

        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            qids.append(pid.view(-1))
            with torch.no_grad():
                cls_txt, _ = model.encode_text(caption)
                text_feats_global.append(cls_txt.cpu())

        for pid, img in self.img_loader:
            img = img.to(device)
            gids.append(pid.view(-1))
            with torch.no_grad():
                cls_img, _ = model.encode_image(img)
                img_feats_global.append(cls_img.cpu())

        return {
            'qids': torch.cat(qids, 0),
            'gids': torch.cat(gids, 0),
            'q_global': torch.cat(text_feats_global, 0),
            'g_global': torch.cat(img_feats_global, 0),
        }

    def _compute_sim_matrix(self, model, features):
        device = next(model.parameters()).device
        q_global = F.normalize(features['q_global'], dim=-1)
        g_global = F.normalize(features['g_global'], dim=-1)
        sims_global = q_global.to(device) @ g_global.to(device).t()
        return sims_global

    def eval(self, model, i2t_metric=True):
        feats = self._extract_all_features(model)
        sims_global = self._compute_sim_matrix(model, feats)
        qids = feats['qids']
        gids = feats['gids']

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
        sims = z_score_norm(sims_global)

        rs = get_metrics(sims, qids, gids, 'Global-t2i', False)
        table.add_row(rs)
        Global_rsum = rs[6]
        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            current_i2t_rsum = i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9]
            table.add_row(['Global-i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP, current_i2t_rsum])
            Global_rsum = rs[6] + current_i2t_rsum

        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        # ... 其余 format 不变
        self.logger.info('\n' + str(table))
        return Global_rsum


    def _compute_sim_matrix(self, model, features):

        device = next(model.parameters()).device
        
        q_global = features['q_global'] 
        g_global = features['g_global'] 
        
        q_global = F.normalize(q_global, dim=-1)
        g_global = F.normalize(g_global, dim=-1)
        
        sims_global = q_global.to(device) @ g_global.to(device).t()

        temp = 0.02
        i2t_probs_g = F.softmax(sims_global / temp, dim=1)
        t2i_probs_g = F.softmax(sims_global / temp, dim=0)
        sims_global_ = i2t_probs_g * t2i_probs_g

                
        # return sims_global_
        return sims_global

    def eval(self, model, i2t_metric=True):
        feats = self._extract_all_features(model)
        sims_global = self._compute_sim_matrix(model, feats)
        qids = feats['qids']
        gids = feats['gids']
        
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
        
        sims_global_norm = z_score_norm(sims_global)

        alpha = 0.9
        sims_dict = {
            'Global': sims_global_norm
        }

        for key in sims_dict.keys():
            sims = sims_dict[key]
            rs = get_metrics(sims, qids, gids, f'{key}-t2i', False)
            table.add_row(rs)
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                current_i2t_rsum = i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9]
                table.add_row([f'{key}-i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP, current_i2t_rsum])
            if key == 'Global':
                Global_rsum = rs[6] + current_i2t_rsum
        
        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["rSum"] = lambda f, v: f"{v:.2f}"
        
        self.logger.info('\n' + str(table)) 

        return Global_rsum
