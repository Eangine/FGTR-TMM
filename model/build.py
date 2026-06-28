from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained
from .hhnc import HHNCCalibrator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class MemoryBank(nn.Module):
    def __init__(self, memory_size, feat_dim):
        super().__init__()
        self.memory_size = memory_size
        self.feat_dim = feat_dim

        self.register_buffer("img_feats", torch.zeros(memory_size, feat_dim))
        self.register_buffer("txt_feats", torch.zeros(memory_size, feat_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("cur_size", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, img_feats, txt_feats):
        if img_feats.numel() == 0 or txt_feats.numel() == 0:
            return

        img_feats = F.normalize(img_feats.detach(), dim=-1)
        txt_feats = F.normalize(txt_feats.detach(), dim=-1)

        num = img_feats.size(0)
        ptr = int(self.ptr.item())

        if num >= self.memory_size:
            self.img_feats.copy_(img_feats[-self.memory_size:])
            self.txt_feats.copy_(txt_feats[-self.memory_size:])
            self.ptr[0] = 0
            self.cur_size[0] = self.memory_size
            return

        end = ptr + num
        if end <= self.memory_size:
            self.img_feats[ptr:end] = img_feats
            self.txt_feats[ptr:end] = txt_feats
        else:
            first = self.memory_size - ptr
            second = num - first
            self.img_feats[ptr:] = img_feats[:first]
            self.img_feats[:second] = img_feats[first:]
            self.txt_feats[ptr:] = txt_feats[:first]
            self.txt_feats[:second] = txt_feats[first:]

        self.ptr[0] = (ptr + num) % self.memory_size
        self.cur_size[0] = min(self.memory_size, int(self.cur_size.item()) + num)

    def is_ready(self, min_size=1):
        return int(self.cur_size.item()) >= min_size

    def get_hhnc_features(self, max_samples=None):
        cur_size = int(self.cur_size.item())
        if cur_size == 0:
            return None, None

        img_feats = self.img_feats[:cur_size]
        txt_feats = self.txt_feats[:cur_size]

        if max_samples is not None and cur_size > max_samples:
            idx = torch.randperm(cur_size, device=img_feats.device)[:max_samples]
            img_feats = img_feats[idx]
            txt_feats = txt_feats[idx]

        return img_feats, txt_feats


class HHNC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._set_task()

        self.use_amp = getattr(args, "use_amp", True)
        self.amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg["embed_dim"]

        temperature = max(float(getattr(args, "temperature", 0.02)), 1e-6)
        self.logit_scale = nn.Parameter(torch.ones([]) / temperature)

        self.use_warmup = True
        self.ignore_gmm = False
        self.use_memory_bank = True
        self.enable_clean_branch = True
        self.memory_size = getattr(args, "memory_size", 4096)
        self.min_bank_size = getattr(args, "min_bank_size", 1024)

        self.memory_bank = MemoryBank(
            memory_size=self.memory_size,
            feat_dim=self.embed_dim,
        )

        self.enable_hhnc = getattr(args, "enable_hhnc", True)
        self.hhnc_loss_weight = getattr(args, "hhnc_loss_weight", 0.1)
        self.hhnc_min_bank_size = getattr(args, "hhnc_min_bank_size", self.min_bank_size)
        self.hhnc_update_interval = max(1, getattr(args, "hhnc_update_interval", 100))
        self.hhnc_max_hierarchy_samples = getattr(args, "hhnc_max_hierarchy_samples", 4096)
        self.hhnc_noisy_ret_weight = getattr(args, "hhnc_noisy_ret_weight", 0.5)
        self.hhnc_step = 0
        self.hhnc_hierarchy = None

        hhnc_fusion_alpha = getattr(args, "hhnc_fusion_alpha", 0.5)
        self.hhnc_calibrator = HHNCCalibrator(
            feature_dim=self.embed_dim,
            hyper_dim=getattr(args, "hhnc_hyper_dim", 256),
            curvature=getattr(args, "hhnc_curvature", 1.0),
            use_hyperbolic=getattr(args, "hhnc_use_hyperbolic", False),
            fusion_alpha=hhnc_fusion_alpha,
            graph_k=getattr(args, "hhnc_graph_k", 16),
            graph_threshold=getattr(args, "hhnc_graph_threshold", 0.55),
            max_microcluster_size=getattr(args, "hhnc_max_microcluster_size", 32),
            kmeans_iters=getattr(args, "hhnc_kmeans_iters", 6),
            min_merge_quality=getattr(args, "hhnc_min_merge_quality", 0.05),
            min_merge_affinity=getattr(args, "hhnc_min_merge_affinity", 0.0),
            quality_lambda_c=getattr(args, "hhnc_quality_lambda_c", 1.0),
            quality_lambda_r=getattr(args, "hhnc_quality_lambda_r", 1.0),
            quality_lambda_d=getattr(args, "hhnc_quality_lambda_d", 1.0),
            reliability_lambda_vt=getattr(args, "hhnc_reliability_lambda_vt", 0.5),
            reliability_lambda_i2t=getattr(args, "hhnc_reliability_lambda_i2t", 0.25),
            reliability_lambda_t2i=getattr(args, "hhnc_reliability_lambda_t2i", 0.25),
            sep_min=getattr(args, "hhnc_sep_min", 0.05),
            sep_max=getattr(args, "hhnc_sep_max", 0.75),
            sep_tau=getattr(args, "hhnc_sep_tau", 0.05),
            topk_lca=getattr(args, "hhnc_topk_lca", 3),
            query_tau=getattr(args, "hhnc_query_tau", 0.07),
            query_conf_threshold=getattr(args, "hhnc_query_conf_threshold", 0.0),
            min_ancestor_depth=getattr(args, "hhnc_min_ancestor_depth", 1),
            parent_margin=getattr(args, "hhnc_parent_margin", 0.5),
            depth_margin=getattr(args, "hhnc_depth_margin", 0.02),
            sibling_margin=getattr(args, "hhnc_sibling_margin", 0.5),
            max_sibling_pairs=getattr(args, "hhnc_max_sibling_pairs", 4096),
        )

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return F.normalize(x[:, 0, :], dim=-1), F.normalize(x[:, 1:, :], dim=-1)

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text)
        eos_idx = text.argmax(dim=-1)
        return F.normalize(x[torch.arange(x.shape[0]), eos_idx], dim=-1), F.normalize(x, dim=-1)

    def _set_task(self):
        self.current_task = [task.strip() for task in self.args.loss_names.split("+")]

    def compute_per_loss(self, batch):
        images = batch["images"]
        caption_ids = batch["caption_ids"]

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            cls_img, _ = self.encode_image(images)
            cls_txt, _ = self.encode_text(caption_ids)
            sims_global = cls_img @ cls_txt.t()
            loss_global, _ = objectives.compute_per_loss(sims_global, batch["pids"], self.args)

        return loss_global.detach().cpu()

    @torch.no_grad()
    def _update_memory_bank(self, clean_mask, cls_img, cls_txt):
        if (not self.use_memory_bank) or clean_mask.sum() == 0:
            return
        self.memory_bank.enqueue(cls_img[clean_mask], cls_txt[clean_mask])

    def _maybe_update_hhnc_hierarchy(self):
        if not self.enable_hhnc:
            return
        if not self.memory_bank.is_ready(self.hhnc_min_bank_size):
            return
        if self.hhnc_hierarchy is not None and (self.hhnc_step % self.hhnc_update_interval) != 0:
            return

        hierarchy_img, hierarchy_txt = self.memory_bank.get_hhnc_features(
            max_samples=self.hhnc_max_hierarchy_samples
        )
        if hierarchy_img is None or hierarchy_txt is None or hierarchy_img.size(0) < 2:
            return

        self.hhnc_hierarchy = self.hhnc_calibrator.build_hierarchy(
            hierarchy_img.detach(),
            hierarchy_txt.detach(),
        )

    def _compute_hhnc_loss(self, cls_img, cls_txt, noisy_mask, sample_prior):
        if (not self.enable_hhnc) or noisy_mask.sum() == 0:
            return None

        self.hhnc_step += 1
        self._maybe_update_hhnc_hierarchy()
        if self.hhnc_hierarchy is None:
            return None

        noisy_weights = sample_prior[noisy_mask].detach()
        if noisy_weights.sum() <= 0:
            return None

        return self.hhnc_calibrator(
            noisy_image_features=cls_img[noisy_mask],
            noisy_text_features=cls_txt[noisy_mask].detach(),
            hierarchy=self.hhnc_hierarchy,
            sample_weights=noisy_weights,
        )

    def _build_global_branch_loss(self, cls_img, cls_txt, pids, args, hist_weights=None, sample_weights=None):
        if cls_img.size(0) == 0:
            return cls_img.sum() * 0.0, None

        global_sims = cls_img @ cls_txt.t()
        return objectives.compute_align_loss(
            global_sims,
            pids,
            args,
            logit_scale=self.logit_scale,
            hist_weights=hist_weights,
            sample_weights=sample_weights,
        )

    def _build_sample_prior(self, label_hat, is_warmup=False):
        prior = torch.ones_like(label_hat, dtype=torch.float32, device=label_hat.device)
        if self.ignore_gmm:
            return prior

        prior[label_hat == 1] = 1.0
        prior[label_hat == 0] = getattr(self.args, "noisy_prior", 0.25)

        if is_warmup:
            prior[label_hat == 0] = 0.0

        return prior

    def forward(self, batch, epoch, args):
        ret = {"temperature": 1 / self.logit_scale}

        images = batch["images"]
        caption_ids = batch["caption_ids"]
        label_hat = batch["label_hat"].to(images.device)

        clean_conf = batch.get("clean_conf", None)
        if clean_conf is not None:
            clean_conf = clean_conf.to(images.device).float().clamp(0.0, 1.0)

        hist_weights = batch.get("hist_weights", None)
        if hist_weights is not None:
            hist_weights = hist_weights.to(images.device)

        if self.ignore_gmm:
            clean_mask = torch.ones_like(label_hat, dtype=torch.bool)
            noisy_mask = torch.zeros_like(label_hat, dtype=torch.bool)
        else:
            clean_mask = label_hat == 1
            noisy_mask = label_hat == 0

        elite_clean_mask = batch.get("elite_clean_mask", None)
        if elite_clean_mask is not None:
            elite_clean_mask = elite_clean_mask.to(images.device).bool() & clean_mask
        else:
            elite_clean_mask = clean_mask

        is_warmup = (
            self.use_warmup
            and getattr(args, "warm_up", False)
            and epoch <= getattr(args, "warmup_epochs", 0)
        )

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            cls_img, _ = self.encode_image(images)
            cls_txt, _ = self.encode_text(caption_ids)

            with torch.no_grad():
                sims_global_orig = cls_img @ cls_txt.t()
                loss_A_per, _ = objectives.compute_per_loss(
                    sims_global_orig,
                    batch["pids"],
                    args,
                    hist_weights=hist_weights,
                    mode="GMM",
                )

            ret["loss_A_per"] = loss_A_per.detach().cpu()

            if is_warmup:
                noisy_mask = torch.zeros_like(noisy_mask)

            sample_prior = self._build_sample_prior(label_hat, is_warmup=is_warmup)

            if self.enable_clean_branch and elite_clean_mask.sum() > 0:
                self._update_memory_bank(elite_clean_mask, cls_img, cls_txt)

            if self.enable_hhnc and noisy_mask.sum() > 0:
                if clean_conf is not None:
                    hhnc_sample_prior = (1.0 - clean_conf).detach()
                    sample_prior[noisy_mask] = clean_conf[noisy_mask].detach() * self.hhnc_noisy_ret_weight
                else:
                    hhnc_sample_prior = sample_prior.clone()
                    sample_prior[noisy_mask] = sample_prior[noisy_mask] * self.hhnc_noisy_ret_weight
            else:
                hhnc_sample_prior = sample_prior.clone()

            align_loss_global, cur_weights_global = self._build_global_branch_loss(
                cls_img,
                cls_txt,
                batch["pids"],
                args,
                hist_weights=hist_weights,
                sample_weights=sample_prior,
            )
            ret["align_loss_global"] = align_loss_global

            hhnc_out = self._compute_hhnc_loss(cls_img, cls_txt, noisy_mask, hhnc_sample_prior)
            if hhnc_out is not None:
                ret.update(
                    {
                        "hhnc_loss": hhnc_out["loss"] * self.hhnc_loss_weight,
                        "hhnc_calibration_value": hhnc_out["calibration_loss"].detach(),
                        "hhnc_hierarchy_value": hhnc_out["hierarchy_loss"].detach(),
                        "hhnc_selected_depth_mean": hhnc_out["selected_depth_mean"].detach(),
                        "hhnc_valid_ratio": hhnc_out["valid_ratio"].detach(),
                        "hhnc_tree_confidence_mean": hhnc_out.get(
                            "tree_confidence_mean",
                            torch.tensor(0.0, device=cls_img.device),
                        ).detach(),
                    }
                )

            if cur_weights_global is not None:
                ret["cur_weights"] = cur_weights_global.detach()

            return ret


def build_model(args):
    return HHNC(args)