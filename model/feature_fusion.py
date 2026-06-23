# # import torch
# # import math
# # import torch.nn as nn
# # import torch.nn.functional as F


# # def mask_xattn_one_text(img_embs, cap_i_expand, img_mask=None, i2t=True, scan=True,):

# #     # (B_v, L_t, L_v)
# #     cap2img_sim = torch.bmm(cap_i_expand, img_embs.transpose(1, 2))

# #     if scan:
# #         cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)

# #     # t2i
# #     # (B_v, L_t)
# #     if img_mask is None:
# #         row_sim = cap2img_sim.max(dim=2)[0]
# #     else:
# #         # Add a low value to the similarity of the masked patch location 
# #         # to prevent it from being selected
# #         row_sim = (cap2img_sim - 1000 * (1 - img_mask).unsqueeze(1)).max(dim=2)[0]  
    
# #     # (B_v, 1)
# #     row_sim_mean = row_sim.mean(dim=1, keepdim=True)

# #     if i2t:
# #         # i2t
# #         # (B_v, L_v)
# #         column_sim = cap2img_sim.max(dim=1)[0]
        
# #         if img_mask is None:
# #             column_sim_mean = column_sim.mean(dim=1, keepdim=True)
# #         else:
# #             # (B_v, 1)
# #             column_sim_mean = (column_sim * img_mask).sum(dim=-1, keepdim=True) / (img_mask.sum(dim=-1, keepdim=True) + 1e-8)

# #         sim_one_text = row_sim_mean + column_sim_mean
# #     else:
# #         sim_one_text = row_sim_mean
    
# #     return sim_one_text 


# # def l2norm(X, dim, eps=1e-8):
# #     """L2-normalize columns of X
# #     """
# #     norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
# #     X = torch.div(X, norm)
# #     return X


# # class LAPS_Vision(nn.Module):
# #     def __init__(self, ratio=0.3):
# #         super(LAPS_Vision, self).__init__()
# #         self.ratio = ratio
    
# #     def get_mask(self, cls_img, patch_img):
# #         """
# #         专门用于论文可视化：返回 Patch Mask (B, Grid, Grid)
# #         Mask = 1.0 (保留/重要), Mask = 0.0 (丢弃/背景)
# #         """
# #         # 1. 准备特征 (与 forward 逻辑完全一致)
# #         img_feats = torch.cat((cls_img.unsqueeze(1), patch_img), dim=1)
# #         img_embs_norm = torch.nn.functional.normalize(img_feats, dim=-1)
        
# #         img_cls_emb_norm = img_embs_norm[:, 0:1, :]
# #         img_spatial_embs = img_feats[:, 1:, :]
# #         img_spatial_embs_norm = img_embs_norm[:, 1:, :]

# #         # 2. 计算分数 (与 forward 逻辑完全一致)
# #         with torch.no_grad():
# #             score_context = (img_cls_emb_norm * img_spatial_embs_norm).sum(dim=-1)
# #             final_score = score_context
        
# #         # 3. 计算保留数量 (使用 eval_ratio，因为我们在做推理)
# #         B_v, L_p, _ = img_spatial_embs.shape
# #         # 这里假设 self.eval_ratio 是保留比例 (例如 0.6 表示保留 60%)
# #         # 如果你的逻辑是反的，请相应调整
# #         num_keep_token = math.ceil(L_p * self.eval_ratio) 

# #         # 4. 获取 Top-K 索引
# #         _, score_index = torch.sort(final_score, dim=1, descending=True)
# #         keep_indices = score_index[:, :num_keep_token] # (B, K)

# #         # 5. 生成 Mask
# #         mask = torch.zeros(B_v, L_p, device=cls_img.device)
# #         mask.scatter_(1, keep_indices, 1.0) # 保留的地方设为1，其他为0

# #         # 6. 还原为 2D 网格 (自动计算 grid size)
# #         grid_size = int(math.sqrt(L_p)) 
# #         mask_2d = mask.view(B_v, grid_size, grid_size)
        
# #         return mask_2d

# #     def forward(self, cls_img, patch_img, cls_txt, token_txt, text, ratio=None):
# #         """
# #         img_feats: (B_v, L_v, C) - 包含 [CLS] at index 0
# #         text_feats: (B_t, L_t, C)
# #         text: (B_t, L_text_tokens)
# #         ratio: 用于控制筛选比例
# #         """

# #         img_feats = torch.cat((cls_img.unsqueeze(1), patch_img), dim=1)
# #         text_feats = torch.cat((cls_txt.unsqueeze(1), token_txt), dim=1)

# #         B_v, L_v, C = img_feats.shape
        
# #         img_embs_norm = F.normalize(img_feats, dim=-1)
# #         cap_embs_norm = F.normalize(token_txt, dim=-1)
# #         img_cls_emb = img_feats[:, 0:1, :] 
# #         img_cls_emb_norm = img_embs_norm[:, 0:1, :]
# #         img_spatial_embs = img_feats[:, 1:, :]
# #         img_spatial_embs_norm = img_embs_norm[:, 1:, :]
      
# #         # 1. Image Context Weight (基于自注意力)
# #         with torch.no_grad():
# #             # (B_v, 1, C) * (B_v, L_p, C) -> (B_v, L_p)
# #             score_context = (img_cls_emb_norm * img_spatial_embs_norm).sum(dim=-1)

# #         final_score = score_context

# #         # --- C. Patch 筛选 (Token Pruning) ---
# #         # num_keep_token = math.ceil((L_v - 1) * 0.98)

# #         if ratio != None:
# #             current_ratio = ratio
# #         else:
# #             current_ratio = self.ratio

# #         num_keep_token = math.ceil((L_v - 1) * current_ratio)
        
# #         # score_index 范围是 [0, L_p-1]
# #         _, score_index = torch.sort(final_score, dim=1, descending=True)
# #         keep_policy = score_index[:, :num_keep_token] # (B_v, K)

# #         select_tokens = torch.gather(img_spatial_embs, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))
# #         fused_tokens = torch.cat([img_cls_emb, select_tokens], dim=1) # (B_v, K+1, C)
# #         fused_tokens = F.normalize(fused_tokens, dim=-1)
# #         improve_sims = []
        
# #         # 这里的循环仅用于计算相似度，不再涉及 Patch 选择逻辑
# #         for i in range(len(text)):
            
# #             n_word = text[i].argmax(dim=-1)
            
# #             # (1, L_t_valid, C) -> (B_v, L_t_valid, C)
# #             cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).expand(B_v, -1, -1)
            
# #             # 计算融合后的图像特征与当前文本的相似度
# #             sim_one_text = mask_xattn_one_text(
# #                 img_embs=fused_tokens,  # 输入的是融合了 CLS 和 Selected Patches 的特征
# #                 cap_i_expand=cap_i_expand,
# #                 scan=True
# #             )
            
# #             improve_sims.append(sim_one_text.t()) # (1, B_v)

# #         # (B_t, B_v)
# #         improve_sims = torch.cat(improve_sims, dim=0)
        
# #         return improve_sims.float()



# # # 代码优化重构
# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F

# def mask_xattn_one_text(img_embs, cap_i, img_mask=None, i2t=True, scan=True):
#     """
#     img_embs: (B_v, L_v, C) - 融合并剪枝后的图像特征
#     cap_i: (L_t_valid, C) - 当前处理的单条有效文本特征
#     """
#     # 【优化】使用 einsum 替代 expand + bmm，完全避免了额外的显存开辟
#     # 结果 shape: (B_v, L_t_valid, L_v)
#     cap2img_sim = torch.einsum('tc,bvc->btv', cap_i, img_embs)

#     if scan:
#         cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)

#     # t2i
#     if img_mask is None:
#         row_sim = cap2img_sim.max(dim=2)[0]
#     else:
#         # Add a low value to the similarity of the masked patch location 
#         # to prevent it from being selected
#         row_sim = (cap2img_sim - 1000 * (1 - img_mask).unsqueeze(1)).max(dim=2)[0]  
    
#     # 【优化】直接降维到 (B_v,) 而不是 (B_v, 1)，方便后续 stack
#     row_sim_mean = row_sim.mean(dim=1)

#     if i2t:
#         # i2t: (B_v, L_v)
#         column_sim = cap2img_sim.max(dim=1)[0]
        
#         if img_mask is None:
#             column_sim_mean = column_sim.mean(dim=1)
#         else:
#             column_sim_mean = (column_sim * img_mask).sum(dim=-1) / (img_mask.sum(dim=-1) + 1e-8)

#         sim_one_text = row_sim_mean + column_sim_mean
#     else:
#         sim_one_text = row_sim_mean
    
#     return sim_one_text 


# class LAPS_Vision(nn.Module):
#     def __init__(self, ratio=0.3):
#         super(LAPS_Vision, self).__init__()
#         self.ratio = ratio
    
#     def get_mask(self, cls_img, patch_img):
#         """
#         专门用于论文可视化：返回 Patch Mask (B, Grid, Grid)
#         Mask = 1.0 (保留/重要), Mask = 0.0 (丢弃/背景)
#         """
#         img_feats = torch.cat((cls_img.unsqueeze(1), patch_img), dim=1)
#         img_embs_norm = F.normalize(img_feats, dim=-1)
        
#         img_cls_emb_norm = img_embs_norm[:, 0:1, :]
#         img_spatial_embs = img_feats[:, 1:, :]
#         img_spatial_embs_norm = img_embs_norm[:, 1:, :]

#         with torch.no_grad():
#             final_score = (img_cls_emb_norm * img_spatial_embs_norm).sum(dim=-1)
        
#         B_v, L_p, _ = img_spatial_embs.shape
        
#         # 【修复 Bug】原代码使用 self.eval_ratio 会报错，此处统一使用 self.ratio 
#         num_keep_token = math.ceil(L_p * self.ratio) 

#         _, score_index = torch.sort(final_score, dim=1, descending=True)
#         keep_indices = score_index[:, :num_keep_token] # (B, K)

#         mask = torch.zeros(B_v, L_p, device=cls_img.device)
#         mask.scatter_(1, keep_indices, 1.0) 

#         grid_size = int(math.sqrt(L_p)) 
#         mask_2d = mask.view(B_v, grid_size, grid_size)
        
#         return mask_2d

#     def forward(self, cls_img, patch_img, cls_txt, token_txt, text, ratio=None):
#         img_feats = torch.cat((cls_img.unsqueeze(1), patch_img), dim=1)
#         text_feats = torch.cat((cls_txt.unsqueeze(1), token_txt), dim=1)

#         B_v, L_v, C = img_feats.shape
        
#         # 全局归一化
#         img_embs_norm = F.normalize(img_feats, dim=-1)
#         cap_embs_norm = F.normalize(token_txt, dim=-1)
        
#         img_cls_emb_norm = img_embs_norm[:, 0:1, :]
#         img_spatial_embs_norm = img_embs_norm[:, 1:, :]
      
#         # 1. Image Context Weight (基于自注意力)
#         with torch.no_grad():
#             final_score = (img_cls_emb_norm * img_spatial_embs_norm).sum(dim=-1)

#         # --- Patch 筛选 (Token Pruning) ---
#         current_ratio = ratio if ratio is not None else self.ratio
#         # 【优化】L_v - 1 本质就是空间 Patch 数量，写法更稳健
#         L_p = img_spatial_embs_norm.shape[1] 
#         num_keep_token = math.ceil(L_p * current_ratio)
        
#         _, score_index = torch.sort(final_score, dim=1, descending=True)
#         keep_policy = score_index[:, :num_keep_token] # (B_v, K)

#         # 【核心优化】直接从已归一化的特征中 gather，省去了重新 normalize 的计算开销
#         select_tokens_norm = torch.gather(
#             img_spatial_embs_norm, 
#             dim=1, 
#             index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
#         )
#         # 拼接后的融合 Token 天生就是已归一化状态，跳过冗余的 F.normalize
#         fused_tokens = torch.cat([img_cls_emb_norm, select_tokens_norm], dim=1)
        
#         improve_sims = []
        
#         # 【优化】将 argmax 提到循环外部，利用向量化计算加速
#         n_words = text.argmax(dim=-1)
        
#         for i in range(len(text)):
#             n_word = n_words[i]
            
#             # 仅截取有效词向量送入函数，无需使用 expand，靠 einsum 完成
#             cap_i = cap_embs_norm[i, :n_word, :] 
            
#             sim_one_text = mask_xattn_one_text(
#                 img_embs=fused_tokens,
#                 cap_i=cap_i,
#                 scan=True
#             )
            
#             improve_sims.append(sim_one_text)

#         # 【优化】(B_t, B_v) 直接通过 stack 生成，跳过每次循环的转置 .t() 与 .cat()
#         improve_sims = torch.stack(improve_sims, dim=0)
        
#         return improve_sims.float()



# 代码优化重构 优化视觉patch选择逻辑，增强对背景干扰的抑制能力
# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F

# class LAPS_Vision(nn.Module):
#     def __init__(self, ratio=0.3):
#         super(LAPS_Vision, self).__init__()
#         self.ratio = ratio

#     def _calculate_hybrid_score(self, img_cls_emb_norm, img_spatial_embs_norm):
#         """
#         计算 CLS 相似度与空间稀有度的联合得分
#         """
#         # --- 1. CLS 相似度得分 ---
#         cls_score = (img_cls_emb_norm * img_spatial_embs_norm).sum(dim=-1) # (B_v, L_p)
#         cls_min = cls_score.min(dim=-1, keepdim=True)[0]
#         cls_max = cls_score.max(dim=-1, keepdim=True)[0]
#         cls_score_norm = (cls_score - cls_min) / (cls_max - cls_min + 1e-8)

#         # --- 2. 新增策略：空间稀有度得分 (【数学级优化】O(L^2) -> O(L)) ---
#         # 提取全局空间均值 (B_v, 1, C)
#         global_spatial_mean = img_spatial_embs_norm.mean(dim=1, keepdim=True)
#         # 直接与均值向量做内积，等效于计算与其他所有Patch的平均相似度
#         avg_sim = (img_spatial_embs_norm * global_spatial_mean).sum(dim=-1) # (B_v, L_p)
        
#         # 取负号：越不相似（稀有）得分越高
#         rarity_score = -avg_sim
#         rarity_min = rarity_score.min(dim=-1, keepdim=True)[0]
#         rarity_max = rarity_score.max(dim=-1, keepdim=True)[0]
#         rarity_score_norm = (rarity_score - rarity_min) / (rarity_max - rarity_min + 1e-8)

#         # --- 3. 联合得分 ---
#         final_score = cls_score_norm + rarity_score_norm
#         return final_score

#     def get_mask(self, cls_img, patch_img):
#         # 此处代码逻辑很好，保持不变（仅调用优化后的 _calculate_hybrid_score）
#         img_feats = torch.cat((cls_img.unsqueeze(1), patch_img), dim=1)
#         img_embs_norm = F.normalize(img_feats, dim=-1)
        
#         img_cls_emb_norm = img_embs_norm[:, 0:1, :]
#         img_spatial_embs_norm = img_embs_norm[:, 1:, :]

#         with torch.no_grad():
#             final_score = self._calculate_hybrid_score(img_cls_emb_norm, img_spatial_embs_norm)
        
#         B_v, L_p = img_spatial_embs_norm.shape[:2]
#         num_keep_token = math.ceil(L_p * self.ratio) 

#         _, score_index = torch.sort(final_score, dim=1, descending=True)
#         keep_indices = score_index[:, :num_keep_token] # (B, K)

#         mask = torch.zeros(B_v, L_p, device=cls_img.device) 
#         mask.scatter_(1, keep_indices, 1.0) 

#         grid_size = int(math.sqrt(L_p)) 
#         mask_2d = mask.view(B_v, grid_size, grid_size)
#         return mask_2d

#     def forward(self, cls_img, patch_img, cls_txt, token_txt, text, ratio=None):
#         img_feats = torch.cat((cls_img.unsqueeze(1), patch_img), dim=1)
#         B_v, L_v_full, C = img_feats.shape
#         B_t, max_L_t = token_txt.shape[:2]
        
#         img_embs_norm = F.normalize(img_feats, dim=-1)
#         cap_embs_norm = F.normalize(token_txt, dim=-1)
        
#         img_cls_emb_norm = img_embs_norm[:, 0:1, :]
#         img_spatial_embs_norm = img_embs_norm[:, 1:, :]
       
#         # 1. 联合打分
#         with torch.no_grad():
#             final_score = self._calculate_hybrid_score(img_cls_emb_norm, img_spatial_embs_norm)

#         # 2. Patch 筛选
#         current_ratio = ratio if ratio is not None else self.ratio
#         L_p = img_spatial_embs_norm.shape[1] 
#         num_keep_token = math.ceil(L_p * current_ratio)
        
#         _, score_index = torch.sort(final_score, dim=1, descending=True)
#         keep_policy = score_index[:, :num_keep_token]

#         select_tokens_norm = torch.gather(
#             img_spatial_embs_norm, 
#             dim=1, 
#             index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
#         )
#         fused_tokens = torch.cat([img_cls_emb_norm, select_tokens_norm], dim=1)
        
#         # --- 3. 【极致提速】全量化交叉注意力，消除 For 循环 ---
#         # text 的 argmax 通常指向 EOS Token，即文本的有效长度
#         seq_lens = text.argmax(dim=-1)
#         # 构建 Text Valid Mask: (B_t, max_L_t)
#         text_mask = torch.arange(max_L_t, device=text.device).unsqueeze(0) <= seq_lens.unsqueeze(1)
        
#         # 计算全局细粒度交互矩阵 -> Shape: (B_t, B_v, max_L_t, L_v_kept)
#         sims = torch.einsum('tic,vjc->tvij', cap_embs_norm, fused_tokens)
#         sims = F.leaky_relu(sims, negative_slope=0.1)
        
#         # --- T2I (Text-to-Image): Max over images, Mean over valid text tokens ---
#         row_sim = sims.max(dim=3)[0]  # (B_t, B_v, max_L_t)
#         row_sim_masked = row_sim * text_mask.unsqueeze(1) # 抹除 Padding 部分的贡献
#         # 取有效文本的平均
#         row_sim_mean = row_sim_masked.sum(dim=2) / (text_mask.sum(dim=1).unsqueeze(1) + 1e-8) # (B_t, B_v)

#         # --- I2T (Image-to-Text): Max over valid texts, Mean over images ---
#         # 防止 Padding 的 Token 在计算 Max 时被选中，赋予一个极小值
#         mask_expanded = text_mask.view(B_t, 1, max_L_t, 1)
#         sims_masked_for_max = sims.masked_fill(~mask_expanded, -1e9)
        
#         col_sim = sims_masked_for_max.max(dim=2)[0] # (B_t, B_v, L_v_kept)
#         col_sim_mean = col_sim.mean(dim=2)          # (B_t, B_v)

#         # 最终得分矩阵
#         improve_sims = row_sim_mean + col_sim_mean  # (B_t, B_v)
        
#         # return improve_sims.float()
#         return improve_sims.float().t()
    










import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class LAPS_Vision(nn.Module):
    def __init__(self, ratio=0.3):
        super(LAPS_Vision, self).__init__()
        self.ratio = ratio

    def _calculate_hybrid_score(self, img_cls_emb_norm, img_spatial_embs_norm):
        """
        计算 CLS 相似度与空间稀有度的联合得分
        """
        # --- 1. CLS 相似度得分 ---
        cls_score = (img_cls_emb_norm * img_spatial_embs_norm).sum(dim=-1) # (B_v, L_p)
        cls_min = cls_score.min(dim=-1, keepdim=True)[0]
        cls_max = cls_score.max(dim=-1, keepdim=True)[0]
        cls_score_norm = (cls_score - cls_min) / (cls_max - cls_min + 1e-8)

        # --- 2. 新增策略：空间稀有度得分 (【数学级优化】O(L^2) -> O(L)) ---
        # 提取全局空间均值 (B_v, 1, C)
        global_spatial_mean = img_spatial_embs_norm.mean(dim=1, keepdim=True)
        # 直接与均值向量做内积，等效于计算与其他所有Patch的平均相似度
        avg_sim = (img_spatial_embs_norm * global_spatial_mean).sum(dim=-1) # (B_v, L_p)
        
        # 取负号：越不相似（稀有）得分越高
        rarity_score = -avg_sim
        rarity_min = rarity_score.min(dim=-1, keepdim=True)[0]
        rarity_max = rarity_score.max(dim=-1, keepdim=True)[0]
        rarity_score_norm = (rarity_score - rarity_min) / (rarity_max - rarity_min + 1e-8)

        # --- 3. 联合得分 ---
        final_score = cls_score_norm + rarity_score_norm
        return final_score

    def get_mask(self, cls_img, patch_img):
        # 此处代码逻辑很好，保持不变（仅调用优化后的 _calculate_hybrid_score）
        img_feats = torch.cat((cls_img.unsqueeze(1), patch_img), dim=1)
        img_embs_norm = F.normalize(img_feats, dim=-1)
        
        img_cls_emb_norm = img_embs_norm[:, 0:1, :]
        img_spatial_embs_norm = img_embs_norm[:, 1:, :]

        with torch.no_grad():
            final_score = self._calculate_hybrid_score(img_cls_emb_norm, img_spatial_embs_norm)
        
        B_v, L_p = img_spatial_embs_norm.shape[:2]
        num_keep_token = math.ceil(L_p * self.ratio) 

        _, score_index = torch.sort(final_score, dim=1, descending=True)
        keep_indices = score_index[:, :num_keep_token] # (B, K)

        mask = torch.zeros(B_v, L_p, device=cls_img.device) 
        mask.scatter_(1, keep_indices, 1.0) 

        grid_size = int(math.sqrt(L_p)) 
        mask_2d = mask.view(B_v, grid_size, grid_size)
        return mask_2d

    def forward(self, cls_img, patch_img, cls_txt, token_txt, text, ratio=None):
        img_feats = torch.cat((cls_img.unsqueeze(1), patch_img), dim=1)
        B_v, L_v_full, C = img_feats.shape
        B_t, max_L_t = token_txt.shape[:2]
        
        img_embs_norm = F.normalize(img_feats, dim=-1)
        cap_embs_norm = F.normalize(token_txt, dim=-1)
        
        img_cls_emb_norm = img_embs_norm[:, 0:1, :]
        img_spatial_embs_norm = img_embs_norm[:, 1:, :]
       
        # 1. 联合打分
        with torch.no_grad():
            final_score = self._calculate_hybrid_score(img_cls_emb_norm, img_spatial_embs_norm)

        # 2. Patch 筛选
        current_ratio = ratio if ratio is not None else self.ratio
        L_p = img_spatial_embs_norm.shape[1] 
        num_keep_token = math.ceil(L_p * current_ratio)
        
        _, score_index = torch.sort(final_score, dim=1, descending=True)
        keep_policy = score_index[:, :num_keep_token]

        select_tokens_norm = torch.gather(
            img_spatial_embs_norm, 
            dim=1, 
            index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )
        fused_tokens = torch.cat([img_cls_emb_norm, select_tokens_norm], dim=1)
        
        

# --- 3. 【极致提速】全量化交叉注意力，消除 For 循环 ---
        seq_lens = text.argmax(dim=-1)
        text_mask = torch.arange(max_L_t, device=text.device).unsqueeze(0) <= seq_lens.unsqueeze(1)
        
        sims = torch.einsum('tic,vjc->tvij', cap_embs_norm, fused_tokens)
        
        # 🚨 优化2：【4D张量显存终极优化】在激活前用 in-place 直接注入大负数 Bias 屏蔽 Padding
        # 彻底避免创建 sims_masked_for_max 这个巨型 4D 副本
        pad_bias = (~text_mask).float() * -1e4
        sims.add_(pad_bias.view(B_t, 1, max_L_t, 1))  # In-place 原地加法
        
        sims = F.leaky_relu(sims, negative_slope=0.1)
        
        # --- T2I (Text-to-Image): Max over images, Mean over valid text tokens ---
        row_sim = sims.max(dim=3)[0]  # (B_t, B_v, max_L_t)
        # 用 0.0 抹除 Padding 的影响以便 sum 取均值
        row_sim_masked = row_sim.masked_fill(~text_mask.unsqueeze(1), 0.0)
        row_sim_mean = row_sim_masked.sum(dim=2) / (text_mask.sum(dim=1).unsqueeze(1) + 1e-8)

        # --- I2T (Image-to-Text): Max over valid texts, Mean over images ---
        # 🚨 因为 padding tokens 已在上面被赋值 -1000 极小值，所以 max 操作天然会无视它们
        col_sim = sims.max(dim=2)[0] # (B_t, B_v, L_v_kept)
        col_sim_mean = col_sim.mean(dim=2)    
        
        improve_sims = row_sim_mean + col_sim_mean  # (B_t, B_v)
        
        # 🚨 手动销毁图节点
        del sims, row_sim, row_sim_masked, col_sim, pad_bias
        
        return improve_sims.float().t()



