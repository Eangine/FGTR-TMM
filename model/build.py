
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.cuda.amp import autocast
from utils.simple_tokenizer import SimpleTokenizer
import torch.distributed as dist
import math

class MemoryBank(nn.Module):
    
    def __init__(self, memory_size, feat_dim, text_len, pad_token_id=0):
        super().__init__()
        self.memory_size = memory_size
        self.feat_dim = feat_dim
        self.text_len = text_len
        self.pad_token_id = pad_token_id

        self.register_buffer("img_feats", torch.zeros(memory_size, feat_dim))
        self.register_buffer("caption_ids", torch.full((memory_size, text_len), pad_token_id, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("cur_size", torch.zeros(1, dtype=torch.long))

        # Python metadata，不参与梯度和 checkpoint
        self.meta = [
            {
                "index": -1,
                "img_path": None,
                "raw_caption": None,
            }
            for _ in range(memory_size)
        ]

    @torch.no_grad()
    def enqueue(self, img_feats, caption_ids, indices=None, img_paths=None, raw_captions=None):
        """
        img_feats: [B, C]
        caption_ids: [B, L]
        indices: list[int] or Tensor[B]
        img_paths: list[str]
        raw_captions: list[str]
        """
        if img_feats.numel() == 0:
            return

        img_feats = F.normalize(img_feats.detach(), dim=-1)
        caption_ids = caption_ids.detach()

        num = img_feats.size(0)
        ptr = int(self.ptr.item())

        if indices is not None and torch.is_tensor(indices):
            indices = indices.detach().cpu().tolist()

        def write_meta(slot, src_idx):
            self.meta[slot] = {
                "index": int(indices[src_idx]) if indices is not None else -1,
                "img_path": img_paths[src_idx] if img_paths is not None else None,
                "raw_caption": raw_captions[src_idx] if raw_captions is not None else None,
            }

        if num >= self.memory_size:
            self.img_feats.copy_(img_feats[-self.memory_size:])
            self.caption_ids.copy_(caption_ids[-self.memory_size:])

            start = num - self.memory_size
            for slot, src_idx in enumerate(range(start, num)):
                write_meta(slot, src_idx)

            self.ptr[0] = 0
            self.cur_size[0] = self.memory_size
            return

        end = ptr + num

        if end <= self.memory_size:
            self.img_feats[ptr:end] = img_feats
            self.caption_ids[ptr:end] = caption_ids

            for j in range(num):
                write_meta(ptr + j, j)
        else:
            first = self.memory_size - ptr
            second = num - first

            self.img_feats[ptr:] = img_feats[:first]
            self.img_feats[:second] = img_feats[first:]

            self.caption_ids[ptr:] = caption_ids[:first]
            self.caption_ids[:second] = caption_ids[first:]

            for j in range(first):
                write_meta(ptr + j, j)
            for j in range(second):
                write_meta(j, first + j)

        self.ptr[0] = (ptr + num) % self.memory_size
        self.cur_size[0] = min(self.memory_size, int(self.cur_size.item()) + num)

    @torch.no_grad()
    def retrieve(self, query_img_feats, topk=5):
        """
        query_img_feats: [B, C]

        return:
            topk_scores: [B, K]
            topk_caption_ids: [B, K, L]
            topk_img_feats: [B, K, C]
            topk_meta: list[list[dict]]
        """
        cur_size = int(self.cur_size.item())
        device = query_img_feats.device

        if cur_size == 0:
            return None, None, None, None

        bank_img = self.img_feats[:cur_size].to(device)
        bank_cap = self.caption_ids[:cur_size].to(device)

        query_img_feats = F.normalize(query_img_feats, dim=-1)
        bank_img = F.normalize(bank_img, dim=-1)

        sims = query_img_feats @ bank_img.t()
        k = min(topk, cur_size)
        topk_scores, topk_idx = sims.topk(k=k, dim=-1)

        topk_caption_ids = bank_cap[topk_idx]
        topk_img_feats = bank_img[topk_idx]

        topk_idx_cpu = topk_idx.detach().cpu().tolist()
        topk_meta = []
        for row in topk_idx_cpu:
            topk_meta.append([self.meta[int(i)] for i in row])

        return topk_scores, topk_caption_ids, topk_img_feats, topk_meta

    def is_ready(self, min_size=1):
        return int(self.cur_size.item()) >= min_size

class RASC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._set_task()

        self.use_amp = getattr(args, 'use_amp', True)
        self.amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg['embed_dim']
        self.tokenizer = SimpleTokenizer()

        self.logit_scale = nn.Parameter(torch.ones([]) * 1 / args.temperature)

        self.mask_token_id = 49405
        self.sos_token_id = 49406
        self.eos_token_id = 49407
        self.pad_token_id = 0

        self.select_ratio = getattr(args, "select_ratio", 0.3)
        self.ntr_chunk_size = getattr(args, "ntr_chunk_size", 1024)
        self.pooling_beta = getattr(args, "pooling_beta", 0.9)
        self.attn_entropy_weight = getattr(args, "attn_entropy_weight", 0.5)
        self.use_selected_patches_for_calibration = getattr(args, "use_selected_patches_for_calibration", True)
        self.global_fusion_lambda = getattr(args, "global_fusion_lambda", 0.5)
        self.reliability_gamma = getattr(args, "reliability_gamma", 1.5)
        self.memory_size = getattr(args, "memory_size", 4096)
        self.retrieval_topk = getattr(args, "retrieval_topk", 5)
        self.min_bank_size = getattr(args, "min_bank_size", 128)
        
        self.ablation_mode = getattr(args, 'ablation_mode', 5)
        self._set_ablation_config()

        text_len = getattr(args, "text_length", 77)
        self.memory_bank = MemoryBank(
            memory_size=self.memory_size,
            feat_dim=self.embed_dim,
            text_len=text_len,
            pad_token_id=self.pad_token_id
        )

        self.enable_visualization = getattr(args,"enable_visualization",False)
        self.vis_dir = getattr(args,"vis_dir","./token_vis")

        self.conf_w_min = getattr(args, "noisy_conf_w_min", 0)   # 置信度最低时权重
        self.conf_w_max = getattr(args, "noisy_conf_w_max", 1)   # 置信度最高时权重
        self.conf_use_logit_scale = getattr(args, "conf_use_logit_scale", True)
        self.conf_use_retrieval_sim = getattr(args, "conf_use_retrieval_sim", True)

    def _set_ablation_config(self):
        """
        mode 0: Baseline
            原始全局对齐。不使用 GMM、不区分 clean/noisy、不使用 memory bank、
            不做 token calibration、不做 noisy 检索。

        mode 1: + GMM 样本加权 / warmup
            使用 GMM (label_hat) 区分 clean/noisy 并生成 sample prior，启用 warmup。
            仍只训练 clean 分支，noisy 样本通过 sample_prior 降权/丢弃。

        mode 2: + Memory bank & noisy 原型检索 (无 token calibration)
            在 mode 1 基础上，clean 样本入 memory bank；
            noisy 样本从 bank 检索 clean 文本原型进行替换。
            但替换后的原型不做 token calibration。

        mode 3: Full model
            在 mode 2 基础上，对 noisy 原型进行 attention-based token calibration
            (加权 token pooling)。
        """

        m = getattr(self.args, 'ablation_mode', 3)

        # -------------------------------------------------------
        # Default switches (= full model)
        # -------------------------------------------------------
        self.ignore_gmm = False
        self.use_warmup = True

        self.enable_clean_branch = True
        self.enable_noisy_branch = True

        self.enable_token_calibration = True
        self.use_attn_support = True

        self.use_salient_patch = getattr(self.args, "use_salient_patch", True)

        self.use_memory_bank = True
        self.use_noisy_retrieval = True
        self.prototype_use_calibration = True

        # -------------------------------------------------------
        # mode 0: baseline (无 GMM / 无 bank / 无校准)
        # -------------------------------------------------------
        if m == 0:
            self.ignore_gmm = True
            self.use_warmup = False

            self.enable_clean_branch = True
            self.enable_noisy_branch = False

            self.enable_token_calibration = False

            self.use_memory_bank = False
            self.use_noisy_retrieval = False
            self.prototype_use_calibration = False

        # -------------------------------------------------------
        # mode 1: + GMM sample prior / warmup
        # -------------------------------------------------------
        elif m == 1:
            self.ignore_gmm = False
            self.use_warmup = True

            self.enable_clean_branch = True
            self.enable_noisy_branch = False

            self.enable_token_calibration = False

            self.use_memory_bank = False
            self.use_noisy_retrieval = False
            self.prototype_use_calibration = False

        # -------------------------------------------------------
        # mode 2: + memory bank & noisy 原型检索替换 (无 calibration)
        # -------------------------------------------------------
        elif m == 2:
            self.ignore_gmm = False
            self.use_warmup = True

            self.enable_clean_branch = True
            self.enable_noisy_branch = True

            # 原型不做 token 校准
            self.enable_token_calibration = False

            self.use_memory_bank = True
            self.use_noisy_retrieval = True
            self.prototype_use_calibration = False

        # -------------------------------------------------------
        # mode 3: full model (+ prototype token calibration)
        # -------------------------------------------------------
        elif m == 5:
            self.ignore_gmm = False
            self.use_warmup = True

            self.enable_clean_branch = True
            self.enable_noisy_branch = True

            self.enable_token_calibration = True   # 用于 noisy prototype 的 token 加权池化

            self.use_memory_bank = True
            self.use_noisy_retrieval = True
            self.prototype_use_calibration = True

        else:
            raise ValueError(f"Unsupported ablation_mode: {m}")

        # -------------------------------------------------------
        # 依赖关系自检 (防止配置出现自相矛盾)
        # -------------------------------------------------------
        if self.enable_noisy_branch:
            assert self.use_memory_bank and self.use_noisy_retrieval, \
                "enable_noisy_branch 需要同时开启 use_memory_bank 与 use_noisy_retrieval"
        if self.use_memory_bank:
            assert self.enable_clean_branch, \
                "use_memory_bank 需要 enable_clean_branch (clean 样本入库)"
        if self.prototype_use_calibration:
            assert self.enable_token_calibration and self.enable_noisy_branch, \
                "prototype_use_calibration 需要 enable_token_calibration 且 enable_noisy_branch"
        if self.ignore_gmm:
            assert not self.enable_noisy_branch, \
                "ignore_gmm=True 时无法区分 noisy，应关闭 enable_noisy_branch"

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return F.normalize(x[:, 0, :], dim=-1), F.normalize(x[:, 1:, :], dim=-1)

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text)
        eos_idx = text.argmax(dim=-1)
        return F.normalize(x[torch.arange(x.shape[0]), eos_idx], dim=-1), F.normalize(x, dim=-1)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]

    def _build_valid_token_mask(self, input_ids):
        valid_mask = (
            (input_ids != self.pad_token_id) &
            (input_ids != self.sos_token_id) &
            (input_ids != self.eos_token_id) &
            (input_ids != self.mask_token_id)
        )
        return valid_mask

    def _calculate_patch_hybrid_score(self, cls_img, patch_img):
        img_feats = torch.cat((cls_img.unsqueeze(1), patch_img), dim=1)
        img_embs_norm = F.normalize(img_feats, dim=-1)

        img_cls_emb_norm = img_embs_norm[:, 0:1, :]     # [B,1,C]
        img_spatial_embs_norm = img_embs_norm[:, 1:, :] # [B,Lp,C]

        cls_score = (img_cls_emb_norm * img_spatial_embs_norm).sum(dim=-1)  # [B, Lp]
        cls_min = cls_score.min(dim=-1, keepdim=True)[0]
        cls_max = cls_score.max(dim=-1, keepdim=True)[0]
        cls_score_norm = (cls_score - cls_min) / (cls_max - cls_min + 1e-8)

        global_spatial_mean = img_spatial_embs_norm.mean(dim=1, keepdim=True)  # [B,1,C]
        avg_sim = (img_spatial_embs_norm * global_spatial_mean).sum(dim=-1)     # [B,Lp]
        rarity_score = -avg_sim
        rarity_min = rarity_score.min(dim=-1, keepdim=True)[0]
        rarity_max = rarity_score.max(dim=-1, keepdim=True)[0]
        rarity_score_norm = (rarity_score - rarity_min) / (rarity_max - rarity_min + 1e-8)

        final_score = cls_score_norm + rarity_score_norm
        return final_score

    def _select_salient_patches(self, cls_img, patch_img, ratio=None, return_idx=False):
        if ratio is None:
            ratio = self.select_ratio

        B, Lp, C = patch_img.shape
        if (not self.use_salient_patch) or ratio >= 1.0:
            if return_idx:
                # 没有剔除：保留全部 patch
                keep_idx = torch.arange(Lp, device=patch_img.device).unsqueeze(0).expand(B, -1)
                full_score = patch_img.new_zeros(B, Lp)
                return patch_img, keep_idx, full_score
            return patch_img

        with torch.no_grad():
            final_score = self._calculate_patch_hybrid_score(cls_img, patch_img)

        num_keep = max(1, math.ceil(Lp * ratio))
        _, score_index = torch.sort(final_score, dim=1, descending=True)
        keep_idx = score_index[:, :num_keep]

        selected_patch = torch.gather(
            patch_img,
            dim=1,
            index=keep_idx.unsqueeze(-1).expand(-1, -1, C)
        )

        if return_idx:
            return selected_patch, keep_idx, final_score
        return selected_patch

    @torch.no_grad()
    def visualize_patch_pruning(self, images, save_prefix="patch_prune",
                                max_samples=8, ratio=None):
        """
        展示视觉 Patch 剔除效果：把保留的 patch 高亮、被剔除的 patch 变暗，
        叠加到原图上保存成对比图。

        images: [B, 3, H, W]，与训练输入一致（已做 normalize）
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        os.makedirs(self.vis_dir, exist_ok=True)
        self.eval()

        B = min(images.size(0), max_samples)
        images = images[:B].to(next(self.parameters()).device)

        cls_img, patch_emb = self.encode_image(images)
        _, keep_idx, final_score = self._select_salient_patches(
            cls_img, patch_emb, ratio=ratio, return_idx=True
        )

        # patch 网格大小：Lp = (H/stride) * (W/stride)
        Lp = patch_emb.size(1)
        H_img, W_img = images.shape[-2], images.shape[-1]
        stride = getattr(self.args, "stride_size", 16)
        gh, gw = H_img // stride, W_img // stride
        # 兼容非整除/含class token的情况，做一次校正
        if gh * gw != Lp:
            side = int(round(math.sqrt(Lp)))
            gh = gw = side

        # 反归一化（CLIP 默认 mean/std）
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=images.device).view(1, 3, 1, 1)
        imgs_denorm = (images * std + mean).clamp(0, 1)

        for b in range(B):
            # 构建 keep mask: [Lp]
            mask = torch.zeros(Lp, device=images.device)
            mask[keep_idx[b]] = 1.0
            mask_grid = mask.reshape(gh, gw).cpu().numpy()

            # 上采样到原图大小
            mask_up = np.kron(mask_grid, np.ones((stride, stride)))
            mask_up = mask_up[:H_img, :W_img]

            img_np = imgs_denorm[b].permute(1, 2, 0).cpu().numpy()

            # 被剔除区域变暗
            dim_factor = 0.25
            overlay = img_np.copy()
            kept = mask_up[..., None]
            overlay = overlay * (kept + (1 - kept) * dim_factor)

            # patch 重要性热力图
            score = final_score[b].reshape(gh, gw).cpu().numpy()
            score = (score - score.min()) / (score.max() - score.min() + 1e-8)
            score_up = np.kron(score, np.ones((stride, stride)))[:H_img, :W_img]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np);            axes[0].set_title("Original")
            axes[1].imshow(score_up, cmap="jet"); axes[1].set_title("Patch Importance")
            axes[2].imshow(overlay)
            axes[2].set_title(f"Kept patches (ratio={ratio or self.select_ratio:.2f})")
            for ax in axes:
                ax.axis("off")

            save_path = os.path.join(self.vis_dir, f"{save_prefix}_{b}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"[PatchPruneVis] saved {B} examples to {self.vis_dir}")

    def _normalize_token_scores(self, score_map, valid_mask, inverse=False, eps=1e-8):
        score_map = score_map.clone().float()
        out = torch.zeros_like(score_map)

        B, L = score_map.shape
        for b in range(B):
            mask = valid_mask[b]
            if mask.sum() == 0:
                continue

            vals = score_map[b][mask]
            vmin = vals.min()
            vmax = vals.max()

            if (vmax - vmin) < eps:
                norm_vals = torch.ones_like(vals) * 0.5
            else:
                norm_vals = (vals - vmin) / (vmax - vmin + eps)

            if inverse:
                norm_vals = 1.0 - norm_vals

            out[b][mask] = norm_vals

        return out

    def _estimate_token_reliability(self, input_ids, cls_img, patch_img):
        """
        基于 token-视觉真实语义匹配度 的文本 token 可靠性估计。
        return:
            token_reliability: [B, L]
            attn_support_map:  [B, L]   (此处即 raw img_match，用于日志诊断)
            valid_mask:        [B, L]
        """
        B, L = input_ids.shape
        device = input_ids.device

        valid_mask = self._build_valid_token_mask(input_ids)

        if valid_mask.sum() == 0:
            zero_map = torch.zeros((B, L), device=device, dtype=torch.float32)
            return zero_map, zero_map, valid_mask

        selected_patch = self._select_salient_patches(
            cls_img,
            patch_img,
            ratio=self.select_ratio if self.use_selected_patches_for_calibration else 1.0
        )

        img_context = torch.cat(
            [cls_img.unsqueeze(1), selected_patch],
            dim=1
        )  # [B, 1 + L_v, C]

        chunk_size = self.ntr_chunk_size

        attn_support_map = torch.zeros((B, L), device=device, dtype=torch.float32)

        with torch.no_grad():
            for start_idx in range(0, B, chunk_size):
                end_idx = min(start_idx + chunk_size, B)

                sub_ids = input_ids[start_idx:end_idx]
                sub_context = img_context[start_idx:end_idx]
                sub_valid_mask = valid_mask[start_idx:end_idx]

                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    _, sub_txt_feats = self.encode_text(sub_ids)

                    t = F.normalize(sub_txt_feats.float(), dim=-1)   # [b, L_text, C]
                    v = F.normalize(sub_context.float(),  dim=-1)    # [b, 1+L_v, C]

                    # token 与每个视觉单元的余弦相似度
                    sim = torch.bmm(t, v.transpose(1, 2))            # [b, L_text, 1+L_v]

                    # 每个 token 在图像中能找到的"最佳匹配度"
                    img_match = sim.max(dim=-1).values               # [b, L_text]

                    conf_attn = img_match.float()
                    conf_attn = conf_attn.masked_fill(~sub_valid_mask, 0.0)

                attn_support_map[start_idx:end_idx] = conf_attn

                del sub_txt_feats, t, v, sim, img_match, conf_attn

        attn_support_map = self._normalize_token_scores(
            attn_support_map,
            valid_mask,
            inverse=False
        )

        token_reliability = attn_support_map

        token_reliability = token_reliability.pow(self.reliability_gamma).clamp(min=0.0, max=1.0)
        token_reliability[~valid_mask] = 0.0

        return token_reliability, attn_support_map, valid_mask
    
    def _weighted_token_pooling(self, cls_txt, token_emb, caption_ids, token_reliability):
        valid_mask = self._build_valid_token_mask(caption_ids).float()
        pooled_weight = self.pooling_beta + (1.0 - self.pooling_beta) * token_reliability
        pooled_weight = pooled_weight * valid_mask
        pooled_txt = (pooled_weight.unsqueeze(-1) * token_emb).sum(dim=1) / (pooled_weight.sum(dim=1, keepdim=True) + 1e-8)
        pooled_txt = F.normalize(pooled_txt, dim=-1)
        fused_txt = self.global_fusion_lambda * cls_txt + (1.0 - self.global_fusion_lambda) * pooled_txt
        fused_txt = F.normalize(fused_txt, dim=-1)

        # ===== 诊断 =====
        # with torch.no_grad():
        #     cos_cls_pooled = (F.normalize(cls_txt, dim=-1) * pooled_txt).sum(-1)   # cls vs pooled
        #     cos_cls_fused  = (F.normalize(cls_txt, dim=-1) * fused_txt).sum(-1)    # cls vs fused
        #     print(f"[Calib] n={cls_txt.size(0)} "
        #         f"cos(cls,pooled)={cos_cls_pooled.mean():.4f} "
        #         f"cos(cls,fused)={cos_cls_fused.mean():.4f} "
        #         f"rel_mean={token_reliability[token_reliability>0].mean():.4f} "
        #         f"rel_std={token_reliability[token_reliability>0].std():.4f}")
        # ================

        return fused_txt, pooled_txt, pooled_weight

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            cls_img, _ = self.encode_image(images)
            cls_txt, _ = self.encode_text(caption_ids)
            sims_global = cls_img @ cls_txt.t()
            loss_global, _ = objectives.compute_per_loss(sims_global, batch['pids'], self.args)

        return loss_global.detach().cpu()

    @torch.no_grad()
    def _update_memory_bank(self, clean_mask, cls_img, caption_ids, batch=None):
        if (not self.use_memory_bank) or clean_mask.sum() == 0:
            return

        indices = None
        img_paths = None
        raw_captions = None

        if batch is not None:
            clean_mask_cpu = clean_mask.detach().cpu().bool().tolist()

            if 'index' in batch:
                all_indices = batch['index'].detach().cpu().tolist()
                indices = [all_indices[i] for i, m in enumerate(clean_mask_cpu) if m]

            if 'img_path' in batch:
                img_paths = [batch['img_path'][i] for i, m in enumerate(clean_mask_cpu) if m]

            if 'raw_caption' in batch:
                raw_captions = [batch['raw_caption'][i] for i, m in enumerate(clean_mask_cpu) if m]

        self.memory_bank.enqueue(
            cls_img[clean_mask],
            caption_ids[clean_mask],
            indices=indices,
            img_paths=img_paths,
            raw_captions=raw_captions
        )

    @torch.no_grad()
    def _retrieve_prototypes(self, query_img_feats, topk=None):
        if topk is None:
            topk = self.retrieval_topk

        if (not self.use_memory_bank) or (not self.memory_bank.is_ready(self.min_bank_size)):
            return None, None, None, None

        topk_scores, topk_caption_ids, topk_img_feats, topk_meta = self.memory_bank.retrieve(
            query_img_feats, topk=topk
        )
        return topk_scores, topk_caption_ids, topk_img_feats, topk_meta

    def _build_global_branch_loss(self, cls_img, cls_txt, pids, args, hist_weights=None, sample_weights=None):
        if cls_img.size(0) == 0:
            return cls_img.sum() * 0.0, None

        global_sims = cls_img @ cls_txt.t()

        loss_global, cur_weights = objectives.compute_align_loss(
            global_sims, pids, args,
            label_hat=None,
            logit_scale=self.logit_scale,
            hist_weights=hist_weights,
            sample_weights=sample_weights
        )
        return loss_global, cur_weights
    
    def _encode_and_calibrate_text(self, caption_ids, cls_img, patch_emb):
        cls_txt, token_emb = self.encode_text(caption_ids)

        if (not self.enable_token_calibration) or caption_ids.size(0) == 0:
            token_weights = torch.ones_like(caption_ids, dtype=torch.float32, device=caption_ids.device)
            return cls_txt, token_emb, token_weights, {}

        token_reliability, attn_support_map, valid_token_mask = self._estimate_token_reliability(
            caption_ids,
            cls_img.detach(),
            patch_emb.detach()
        )

        fused_txt, pooled_txt, calibrated_weights = self._weighted_token_pooling(cls_txt, token_emb, caption_ids, token_reliability)

        aux = {
            'token_reliability': token_reliability.detach(),
            'attn_support_map': attn_support_map.detach(),
            'token_weights': calibrated_weights.detach(),
        }
        return fused_txt, token_emb, calibrated_weights, aux

    def _select_best_prototype(self, query_cls_img, query_patch_emb, proto_caption_ids):
        """
        从 top-k prototype 中为每个 noisy 样本选出一个最佳文本原型。

        return: 
            best_caption_ids: [B, L]
            best_cls_txt: [B, C]
            best_token_emb: [B, L, C]
            best_idx: [B]
            global_scores: [B, K]
        """
        B, K, L = proto_caption_ids.shape
        device = query_cls_img.device

        if K == 1:
            best_caption_ids = proto_caption_ids.squeeze(1)
            best_cls_txt, best_token_emb = self.encode_text(best_caption_ids)

            best_idx = torch.zeros(B, dtype=torch.long, device=device)
            global_scores = torch.ones(B, 1, dtype=query_cls_img.dtype, device=device)

            return best_caption_ids, best_cls_txt, best_token_emb, best_idx, global_scores

        flat_proto_ids = proto_caption_ids.reshape(B * K, L)
        proto_cls_txt, proto_token_emb = self.encode_text(flat_proto_ids)

        proto_cls_txt = proto_cls_txt.view(B, K, -1)
        proto_token_emb = proto_token_emb.view(B, K, L, -1)

        global_scores = torch.einsum('bc,bkc->bk', query_cls_img, proto_cls_txt)
        best_idx = global_scores.argmax(dim=1)
        batch_idx = torch.arange(B, device=device)

        best_caption_ids = proto_caption_ids[batch_idx, best_idx]
        best_cls_txt = proto_cls_txt[batch_idx, best_idx]
        best_token_emb = proto_token_emb[batch_idx, best_idx]

        return best_caption_ids, best_cls_txt, best_token_emb, best_idx, global_scores

    def _compute_retrieval_confidence(self, global_scores, topk_scores=None):
        """
        基于图-候选文本相似度分布的熵，计算伪文本置信度权重。

        global_scores: [B, K]  query 图片与 K 个候选文本的余弦相似度 (image-text)
        topk_scores:   [B, K]  query 图片与 K 个候选图片的相似度 (image-image)

        思想：
        - 用 logit_scale 作为逆温度，把 K 个相似度变为分布 p
        - 归一化熵 H_norm = H(p) / log(K) ∈ [0,1]，熵低 -> 选择确定 -> 置信度高
        - confidence = 1 - H_norm，可再与图-图检索相似度联合
        - 线性映射到 [w_min, w_max]
        return:
        conf_weight: [B]
        """
        B, K = global_scores.shape
        device = global_scores.device

        if K <= 1:
            return torch.full((B,), self.conf_w_max, device=device, dtype=torch.float32)

        if self.conf_use_logit_scale:
            logits = global_scores.float() * self.logit_scale
        else:
            tau = getattr(self.args, "conf_entropy_tau", 0.02)
            logits = global_scores.float() / tau

        p = F.softmax(logits, dim=-1)                          # [B, K]

        eps = 1e-8
        entropy = -(p * (p + eps).log()).sum(dim=-1)           # [B]
        norm_entropy = entropy / math.log(K)                   # [B] ∈ [0,1]

        confidence = (1.0 - norm_entropy).clamp(0.0, 1.0)      # 熵低 -> 置信度高

        # 联合图-图检索相似度：检索来源越相似，伪文本越可信
        if topk_scores is not None and self.conf_use_retrieval_sim:
            retr_sim = topk_scores.float().mean(dim=-1).clamp(min=0.0)  # [B]
            confidence = confidence * retr_sim

        conf_weight = self.conf_w_min + (self.conf_w_max - self.conf_w_min) * confidence
        conf_weight = conf_weight.clamp(self.conf_w_min, self.conf_w_max)
        return conf_weight

    def _build_sample_prior(self, label_hat, is_warmup=False):
        """
        根据 GMM 标签生成样本级先验权重
        clean=1.0, noisy=0.25
        warmup 时可只保留 clean / 或降低其余权重
        """
        prior = torch.ones_like(label_hat, dtype=torch.float32, device=label_hat.device)

        if self.ignore_gmm:
            return prior

        prior[label_hat == 1] = 1.0
        prior[label_hat == 0] = getattr(self.args, "noisy_prior", 0.25)

        if is_warmup:
            prior[label_hat == 0] = 0.0

        return prior

    def _decode_caption(self, token_ids):
        """
        将 CLIP token ids 转成可读文本。
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().tolist()

        valid_ids = []
        for t in token_ids:
            t = int(t)
            if t in [self.pad_token_id, self.sos_token_id, self.eos_token_id, self.mask_token_id]:
                continue
            valid_ids.append(t)

        if len(valid_ids) == 0:
            return ""

        try:
            text = self.tokenizer.decode(valid_ids)
            text = text.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
            return text.strip()
        except Exception:
            words = []
            for t in valid_ids:
                if t in self.tokenizer.decoder:
                    words.append(self.tokenizer.decoder[t])
            return "".join(words).replace("</w>", " ").strip()

    def _denorm_image_np(self, image_tensor):
        """
        image_tensor: [3,H,W], normalized by CLIP mean/std
        return: uint8 RGB image
        """
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=image_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=image_tensor.device).view(3, 1, 1)

        img = (image_tensor * std + mean).clamp(0, 1)
        img = img.detach().cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype("uint8")
        return img

    @torch.no_grad()
    def _save_patch_masked_image(self, image_tensor, cls_img_one, patch_emb_one, save_path):
        """
        保存 patch pruning 后的可视化图片。
        image_tensor: [3,H,W]
        cls_img_one: [1,C]
        patch_emb_one: [1,Lp,C]
        """
        import numpy as np
        from PIL import Image

        device = image_tensor.device
        _, keep_idx, final_score = self._select_salient_patches(
            cls_img_one,
            patch_emb_one,
            ratio=self.select_ratio,
            return_idx=True
        )

        Lp = patch_emb_one.size(1)
        H_img, W_img = image_tensor.shape[-2], image_tensor.shape[-1]
        stride = getattr(self.args, "stride_size", 16)

        gh, gw = H_img // stride, W_img // stride
        if gh * gw != Lp:
            side = int(round(math.sqrt(Lp)))
            gh = gw = side

        mask = torch.zeros(Lp, device=device)
        mask[keep_idx[0]] = 1.0
        mask_grid = mask.reshape(gh, gw).detach().cpu().numpy()

        mask_up = np.kron(mask_grid, np.ones((stride, stride)))
        mask_up = mask_up[:H_img, :W_img]

        img_np = self._denorm_image_np(image_tensor)
        img_float = img_np.astype("float32") / 255.0

        dim_factor = 0.25
        kept = mask_up[..., None]
        overlay = img_float * (kept + (1 - kept) * dim_factor)
        overlay = (overlay * 255).clip(0, 255).astype("uint8")

        Image.fromarray(overlay).save(save_path)

    @torch.no_grad()
    def _save_noisy_case_visualization(
        self,
        batch,
        vis_request,
        images,
        cls_img,
        patch_emb,
        noisy_mask,
        topk_scores,
        proto_caption_ids,
        topk_meta,
        best_caption_ids,
        best_idx,
        conf_weight,
        token_reliability=None
    ):
        """
        保存 noisy 样本案例：
        - 原图
        - patch pruning 掩码图
        - 原始噪声文本
        - 检索到的图片和文本
        - 最终校准文本
        - 文本可信度
        """
        import os
        import json
        import shutil
        from PIL import Image

        if vis_request is None:
            return

        save_root = vis_request.get("save_dir", self.vis_dir)
        epoch = vis_request.get("epoch", -1)
        iteration = vis_request.get("iter", -1)
        max_cases = vis_request.get("max_cases", 4)
        token_thr = vis_request.get("token_conf_thr", 0.3)

        os.makedirs(save_root, exist_ok=True)

        noisy_indices_in_batch = torch.where(noisy_mask)[0].detach().cpu().tolist()
        if len(noisy_indices_in_batch) == 0:
            return

        num_save = min(len(noisy_indices_in_batch), max_cases)

        for local_noisy_id in range(num_save):
            bidx = noisy_indices_in_batch[local_noisy_id]

            global_index = None
            if 'index' in batch:
                global_index = int(batch['index'][bidx].detach().cpu().item())
            else:
                global_index = int(bidx)

            case_dir = os.path.join(
                save_root,
                f"epoch{epoch:03d}_iter{iteration:06d}_idx{global_index}"
            )
            os.makedirs(case_dir, exist_ok=True)

            # -----------------------------
            # 1. 保存原始图片
            # -----------------------------
            img_path = None
            if 'img_path' in batch:
                img_path = batch['img_path'][bidx]

            original_save_path = os.path.join(case_dir, "original_image.jpg")

            if img_path is not None and os.path.exists(img_path):
                try:
                    shutil.copy2(img_path, original_save_path)
                except Exception:
                    Image.fromarray(self._denorm_image_np(images[bidx])).save(original_save_path)
            else:
                Image.fromarray(self._denorm_image_np(images[bidx])).save(original_save_path)

            # -----------------------------
            # 2. 保存 patch mask 后的图片
            # -----------------------------
            masked_save_path = os.path.join(case_dir, "masked_image.jpg")
            self._save_patch_masked_image(
                images[bidx],
                cls_img[bidx:bidx + 1],
                patch_emb[bidx:bidx + 1],
                masked_save_path
            )

            # -----------------------------
            # 3. 原始噪声文本
            # -----------------------------
            noisy_text = None
            if 'raw_caption' in batch:
                noisy_text = batch['raw_caption'][bidx]
            else:
                noisy_text = self._decode_caption(batch['caption_ids'][bidx])

            # -----------------------------
            # 4. 检索结果
            # -----------------------------
            retrieved_infos = []
            K = proto_caption_ids.size(1)

            for k in range(K):
                meta = topk_meta[local_noisy_id][k] if topk_meta is not None else {}

                retrieved_text = meta.get("raw_caption", None)
                if retrieved_text is None:
                    retrieved_text = self._decode_caption(proto_caption_ids[local_noisy_id, k])

                retrieved_img_path = meta.get("img_path", None)
                retrieved_img_save = None

                if retrieved_img_path is not None and os.path.exists(retrieved_img_path):
                    retrieved_img_save = os.path.join(case_dir, f"retrieved_top{k + 1}.jpg")
                    try:
                        shutil.copy2(retrieved_img_path, retrieved_img_save)
                    except Exception:
                        retrieved_img_save = None

                retrieved_infos.append({
                    "rank": k + 1,
                    "bank_index": meta.get("index", -1),
                    "image_path": retrieved_img_path,
                    "saved_image": retrieved_img_save,
                    "text": retrieved_text,
                    "image_image_similarity": float(topk_scores[local_noisy_id, k].detach().cpu().item()),
                    "is_selected": int(k == int(best_idx[local_noisy_id].detach().cpu().item()))
                })

            # -----------------------------
            # 5. 最终校准文本和可信度
            # -----------------------------
            final_text = self._decode_caption(best_caption_ids[local_noisy_id])
            text_confidence = float(conf_weight[local_noisy_id].detach().cpu().item())

            token_conf_list = []
            reliable_text = final_text

            if token_reliability is not None:
                ids = best_caption_ids[local_noisy_id].detach().cpu().tolist()
                rel = token_reliability[local_noisy_id].detach().cpu().tolist()

                reliable_token_ids = []

                for tid, score in zip(ids, rel):
                    tid = int(tid)
                    if tid in [self.pad_token_id, self.sos_token_id, self.eos_token_id, self.mask_token_id]:
                        continue

                    token_str = ""
                    try:
                        token_str = self.tokenizer.decode([tid]).strip()
                    except Exception:
                        token_str = self.tokenizer.decoder.get(tid, "")

                    token_conf_list.append({
                        "token_id": tid,
                        "token": token_str,
                        "confidence": float(score)
                    })

                    if float(score) >= token_thr:
                        reliable_token_ids.append(tid)

                if len(reliable_token_ids) > 0:
                    reliable_text = self._decode_caption(reliable_token_ids)

            # -----------------------------
            # 6. 保存 JSON
            # -----------------------------
            result = {
                "epoch": epoch,
                "iteration": iteration,
                "global_index": global_index,
                "original_image_path": img_path,
                "original_image_saved": original_save_path,
                "masked_image_saved": masked_save_path,
                "original_noisy_text": noisy_text,
                "retrieved_candidates": retrieved_infos,
                "selected_rank": int(best_idx[local_noisy_id].detach().cpu().item()) + 1,
                "final_calibrated_text": final_text,
                "reliable_text_by_token_threshold": reliable_text,
                "text_confidence": text_confidence,
                "token_confidence": token_conf_list
            }

            with open(os.path.join(case_dir, "case_info.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)


    def forward(self, batch, epoch, args):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        vis_request = batch.get("vis_request", None)
        enable_vis = (
            getattr(self, "enable_visualization", False)
            and isinstance(vis_request, dict)
            and len(vis_request) > 0
        )
        vis_data = {}

        images = batch['images']
        caption_ids = batch['caption_ids']
        label_hat = batch['label_hat'].to(images.device)

        # 尝试从 batch 中获取全局大数组传进来的历史权重 w_i^{(t-1)}
        hist_weights = batch.get('hist_weights', None)
        if hist_weights is not None:
            hist_weights = hist_weights.to(images.device)

        if self.ignore_gmm:
            clean_mask = torch.ones_like(label_hat, dtype=torch.bool)
            noisy_mask = torch.zeros_like(label_hat, dtype=torch.bool)
        else:
            clean_mask = (label_hat == 1)
            noisy_mask = (label_hat == 0) 

        is_warmup = (self.use_warmup and args.warm_up and epoch <= args.warmup_epochs)
    
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            cls_img, patch_emb = self.encode_image(images)
            cls_txt, token_emb = self.encode_text(caption_ids)

            # 1) 只记录全局损失给 GMM
            with torch.no_grad():
                sims_global_orig = cls_img @ cls_txt.t()
                loss_A_per, _ = objectives.compute_per_loss(
                    sims_global_orig, batch['pids'], args,
                    hist_weights=hist_weights, mode='GMM'
                )

            ret.update({'loss_A_per': loss_A_per.detach().cpu()})
            # 不再有 loss_B_per

            if is_warmup:
                noisy_mask = torch.zeros_like(noisy_mask)

            B = images.size(0)
            device = images.device

            fused_cls_txt_all = cls_txt.clone()
            sample_prior = self._build_sample_prior(label_hat, is_warmup=is_warmup)

            # 2.1 clean: 入 memory bank
            if self.enable_clean_branch and clean_mask.sum() > 0:
                self._update_memory_bank(clean_mask, cls_img, caption_ids, batch=batch)

            # 2.2 noisy: 检索 + 原型校准
            if self.enable_noisy_branch and noisy_mask.sum() > 0 and self.use_noisy_retrieval and sample_prior[noisy_mask].sum() > 0:
                noisy_cls_img = cls_img[noisy_mask]
                noisy_patch_emb = patch_emb[noisy_mask]

                topk_scores, proto_caption_ids, _, topk_meta = self._retrieve_prototypes(
                    noisy_cls_img, topk=self.retrieval_topk
                )

                if proto_caption_ids is not None:
                    best_caption_ids, best_cls_txt, best_token_emb, best_idx, rerank_scores = self._select_best_prototype(
                        noisy_cls_img, noisy_patch_emb, proto_caption_ids
                    )

                    if self.conf_use_retrieval_sim:
                        conf_weight = self._compute_retrieval_confidence(
                            rerank_scores, topk_scores=topk_scores
                        )
                        sample_prior[noisy_mask] = conf_weight.to(sample_prior.dtype)
                        ret.update({'noisy_conf_weight_mean': conf_weight.mean().detach().cpu()})
                    else:
                        conf_weight = sample_prior[noisy_mask].detach().float()

                    proto_token_reliability = None

                    if self.prototype_use_calibration and self.enable_token_calibration:
                        noisy_fused_cls_txt, _, _, proto_aux = self._encode_and_calibrate_text(
                            best_caption_ids, noisy_cls_img, noisy_patch_emb
                        )
                        if isinstance(proto_aux, dict) and 'token_reliability' in proto_aux:
                            proto_token_reliability = proto_aux['token_reliability']
                    else:
                        noisy_fused_cls_txt = best_cls_txt

                    fused_cls_txt_all[noisy_mask] = noisy_fused_cls_txt

                    # ============================
                    # Case-level visualization
                    # ============================
                    if enable_vis:
                        self._save_noisy_case_visualization(
                            batch=batch,
                            vis_request=vis_request,
                            images=images.detach(),
                            cls_img=cls_img.detach(),
                            patch_emb=patch_emb.detach(),
                            noisy_mask=noisy_mask.detach(),
                            topk_scores=topk_scores.detach(),
                            proto_caption_ids=proto_caption_ids.detach(),
                            topk_meta=topk_meta,
                            best_caption_ids=best_caption_ids.detach(),
                            best_idx=best_idx.detach(),
                            conf_weight=conf_weight.detach(),
                            token_reliability=proto_token_reliability.detach() if proto_token_reliability is not None else None
                        )

                else:
                    sample_prior[noisy_mask] = 0.0


            # 3) 只做全局对齐
            align_loss_global, cur_weights_global = self._build_global_branch_loss(
                cls_img,
                fused_cls_txt_all,
                batch['pids'],
                args,
                hist_weights=hist_weights,
                sample_weights=sample_prior
            )

            ret.update({'align_loss_global': align_loss_global})

            if cur_weights_global is not None:
                ret.update({'cur_weights': cur_weights_global.detach()})
            
            return ret



def build_model(args):
    model = RASC(args)
    return model

