import heapq
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


def l2_normalize(x):
    return F.normalize(x, dim=-1)


def ball_clip(x, eps=1e-5):
    max_norm = 1.0 - eps
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def expmap0(x, c=1.0):
    sqrt_c = c ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    y = torch.tanh(sqrt_c * norm) * x / (sqrt_c * norm)
    return ball_clip(y)


def poincare_distance(x, y, c=1.0):
    x = ball_clip(x)
    y = ball_clip(y)
    x2 = (x * x).sum(dim=-1).clamp(max=1.0 - EPS)
    y2 = (y * y).sum(dim=-1).clamp(max=1.0 - EPS)
    diff2 = ((x - y) * (x - y)).sum(dim=-1)
    denom = (1.0 - x2) * (1.0 - y2)
    z = 1.0 + 2.0 * c * diff2 / denom.clamp_min(EPS)
    return torch.acosh(z.clamp_min(1.0 + EPS))


class HyperbolicProjector(nn.Module):
    def __init__(self, in_dim, out_dim, curvature=1.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.curvature = curvature

    def forward(self, x):
        return expmap0(self.proj(x), c=self.curvature)


class HHNCCalibrator(nn.Module):
    """Quality-aware hierarchical consensus calibration.

    The core calibration path is Euclidean by default. Hyperbolic hierarchy
    regularization is enabled only when ``use_hyperbolic`` is true.
    """

    def __init__(
        self,
        feature_dim,
        hyper_dim=256,
        curvature=1.0,
        use_hyperbolic=False,
        fusion_alpha=0.5,
        graph_k=16,
        graph_threshold=0.55,
        max_microcluster_size=32,
        kmeans_iters=6,
        min_merge_quality=0.05,
        min_merge_affinity=0.0,
        quality_lambda_c=1.0,
        quality_lambda_r=1.0,
        quality_lambda_d=1.0,
        reliability_lambda_vt=0.5,
        reliability_lambda_i2t=0.25,
        reliability_lambda_t2i=0.25,
        sep_min=0.05,
        sep_max=0.75,
        sep_tau=0.05,
        topk_lca=3,
        query_tau=0.07,
        query_conf_threshold=0.0,
        min_ancestor_depth=1,
        parent_margin=0.5,
        depth_margin=0.02,
        sibling_margin=0.5,
        max_sibling_pairs=4096,
    ):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature
        self.fusion_alpha = fusion_alpha
        self.graph_k = graph_k
        self.graph_threshold = graph_threshold
        self.max_microcluster_size = max_microcluster_size
        self.kmeans_iters = kmeans_iters
        self.min_merge_quality = min_merge_quality
        self.min_merge_affinity = min_merge_affinity
        self.quality_lambda_c = quality_lambda_c
        self.quality_lambda_r = quality_lambda_r
        self.quality_lambda_d = quality_lambda_d
        self.reliability_lambda_vt = reliability_lambda_vt
        self.reliability_lambda_i2t = reliability_lambda_i2t
        self.reliability_lambda_t2i = reliability_lambda_t2i
        self.sep_min = sep_min
        self.sep_max = sep_max
        self.sep_tau = sep_tau
        self.topk_lca = topk_lca
        self.query_tau = query_tau
        self.query_conf_threshold = query_conf_threshold
        self.min_ancestor_depth = min_ancestor_depth
        self.parent_margin = parent_margin
        self.depth_margin = depth_margin
        self.sibling_margin = sibling_margin
        self.max_sibling_pairs = max_sibling_pairs

        if use_hyperbolic:
            self.image_projector = HyperbolicProjector(feature_dim, hyper_dim, curvature)
            self.text_projector = HyperbolicProjector(feature_dim, hyper_dim, curvature)
            self.node_projector = HyperbolicProjector(feature_dim, hyper_dim, curvature)
        else:
            self.image_projector = None
            self.text_projector = None
            self.node_projector = None

    @torch.no_grad()
    def build_hierarchy(self, image_features, text_features):
        image_features = l2_normalize(image_features.float())
        text_features = l2_normalize(text_features.float())
        num_samples = image_features.size(0)
        if num_samples < 2:
            return None

        fused_features = l2_normalize(
            self.fusion_alpha * text_features + (1.0 - self.fusion_alpha) * image_features
        )

        affinity, mutual_mask = self._build_affinity_graph(
            image_features, text_features, fused_features
        )
        init_mask = mutual_mask & (affinity > self.graph_threshold)
        components = self._connected_components(init_mask)
        components = self._split_oversized_components(components, fused_features)
        if not components:
            return None

        nodes = []
        sample_to_node = torch.full((num_samples,), -1, dtype=torch.long, device=image_features.device)
        for samples in components:
            node_id = len(nodes)
            nodes.append(
                self._make_node(
                    samples=samples,
                    image_features=image_features,
                    text_features=text_features,
                    fused_features=fused_features,
                    children=[],
                    virtual=False,
                    quality_override=None,
                    sep_value=1.0,
                )
            )
            sample_to_node[samples] = node_id

        leaf_ids = list(range(len(nodes)))
        active = set(leaf_ids)
        heap = []
        candidate_pairs = self._initial_candidate_pairs(mutual_mask, sample_to_node)
        for a, b in candidate_pairs:
            self._push_candidate(
                heap, nodes, a, b, affinity, image_features, text_features, fused_features
            )

        while len(active) > 1 and heap:
            neg_score, _, a, b = heapq.heappop(heap)
            if a not in active or b not in active:
                continue

            score, quality, _ = self._merge_score(
                nodes[a], nodes[b], affinity, image_features, text_features, fused_features
            )
            if score <= 0.0 or quality < self.min_merge_quality:
                continue
            if abs(score + neg_score) > 1e-4:
                self._push_candidate(
                    heap, nodes, a, b, affinity, image_features, text_features, fused_features
                )
                continue

            merged_samples = torch.cat([nodes[a]["samples"], nodes[b]["samples"]], dim=0)
            parent_id = len(nodes)
            parent_node = self._make_node(
                samples=merged_samples,
                image_features=image_features,
                text_features=text_features,
                fused_features=fused_features,
                children=[a, b],
                virtual=False,
                quality_override=None,
                sep_value=None,
                child_a=nodes[a],
                child_b=nodes[b],
            )
            nodes.append(parent_node)
            nodes[a]["parent"] = parent_id
            nodes[b]["parent"] = parent_id
            active.remove(a)
            active.remove(b)
            active.add(parent_id)

            for other in list(active):
                if other == parent_id:
                    continue
                self._push_candidate(
                    heap, nodes, parent_id, other, affinity,
                    image_features, text_features, fused_features
                )

        virtual_root = -1
        if len(active) > 1:
            root_samples = torch.arange(num_samples, device=image_features.device)
            virtual_root = len(nodes)
            root = self._make_node(
                samples=root_samples,
                image_features=image_features,
                text_features=text_features,
                fused_features=fused_features,
                children=sorted(active),
                virtual=True,
                quality_override=0.0,
                sep_value=1.0,
            )
            nodes.append(root)
            for child in active:
                nodes[child]["parent"] = virtual_root
            root_id = virtual_root
        else:
            root_id = next(iter(active))

        parent, depth = self._finalize_depth(nodes, root_id)
        sibling_pairs = self._make_sibling_pairs(nodes, parent, virtual_root, image_features.device)

        node_features = torch.stack([n["cr"] for n in nodes], dim=0)
        node_image_features = torch.stack([n["cv"] for n in nodes], dim=0)
        node_text_features = torch.stack([n["ct"] for n in nodes], dim=0)
        node_quality = torch.tensor([n["quality"] for n in nodes], device=image_features.device)
        leaves = torch.tensor(leaf_ids, dtype=torch.long, device=image_features.device)
        leaf_paths, leaf_path_mask = self._make_leaf_paths(parent, leaves)

        return {
            "parent": parent,
            "depth": depth,
            "node_features": node_features,
            "node_image_features": node_image_features,
            "node_text_features": node_text_features,
            "node_quality": node_quality.clamp(0.0, 1.0),
            "leaves": leaves,
            "leaf_features": node_features[leaves],
            "leaf_paths": leaf_paths,
            "leaf_path_mask": leaf_path_mask,
            "sibling_pairs": sibling_pairs,
            "virtual_root": torch.tensor(virtual_root, dtype=torch.long, device=image_features.device),
        }

    def _build_affinity_graph(self, image_features, text_features, fused_features):
        sim = (
            0.5 * (fused_features @ fused_features.t())
            + 0.25 * (image_features @ image_features.t())
            + 0.25 * (text_features @ text_features.t())
        )
        affinity = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
        affinity.fill_diagonal_(0.0)

        n = affinity.size(0)
        k = min(max(1, self.graph_k), max(1, n - 1))
        knn_idx = affinity.topk(k=k, dim=1).indices
        knn_mask = torch.zeros((n, n), dtype=torch.bool, device=affinity.device)
        knn_mask.scatter_(1, knn_idx, True)
        mutual_mask = knn_mask & knn_mask.t()
        return affinity, mutual_mask

    def _connected_components(self, edge_mask):
        edge_cpu = edge_mask.detach().cpu()
        n = edge_cpu.size(0)
        seen = torch.zeros(n, dtype=torch.bool)
        components = []

        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            comp = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                nbrs = torch.nonzero(edge_cpu[cur], as_tuple=False).flatten().tolist()
                for nb in nbrs:
                    if not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
            components.append(torch.tensor(comp, dtype=torch.long, device=edge_mask.device))
        return components

    def _split_oversized_components(self, components, fused_features):
        out = []
        for comp in components:
            if comp.numel() <= self.max_microcluster_size:
                out.append(comp)
                continue
            parts = int(math.ceil(comp.numel() / float(self.max_microcluster_size)))
            out.extend(self._spherical_kmeans_split(comp, fused_features, parts))
        return out

    def _spherical_kmeans_split(self, samples, features, num_parts):
        feats = features[samples]
        n = feats.size(0)
        if n <= 1 or num_parts <= 1:
            return [samples]

        num_parts = min(num_parts, n)
        init_idx = torch.linspace(0, n - 1, steps=num_parts, device=features.device).long()
        centroids = feats[init_idx].clone()
        assignment = torch.zeros(n, dtype=torch.long, device=features.device)

        for _ in range(self.kmeans_iters):
            scores = feats @ l2_normalize(centroids).t()
            assignment = scores.argmax(dim=1)
            for k in range(num_parts):
                mask = assignment == k
                if mask.any():
                    centroids[k] = feats[mask].mean(dim=0)

        parts = []
        for k in range(num_parts):
            mask = assignment == k
            if mask.any():
                parts.append(samples[mask])
        if not parts:
            parts = list(torch.chunk(samples, num_parts))
        return [p for p in parts if p.numel() > 0]

    def _make_node(
        self,
        samples,
        image_features,
        text_features,
        fused_features,
        children,
        virtual,
        quality_override,
        sep_value,
        child_a=None,
        child_b=None,
    ):
        cv = l2_normalize(image_features[samples].mean(dim=0, keepdim=True)).squeeze(0)
        ct = l2_normalize(text_features[samples].mean(dim=0, keepdim=True)).squeeze(0)
        cr = l2_normalize(fused_features[samples].mean(dim=0, keepdim=True)).squeeze(0)

        coherence = (fused_features[samples] @ cr).mean().clamp(-1.0, 1.0)
        reliability = self._cluster_reliability(samples, cv, ct, image_features, text_features)

        if sep_value is None and child_a is not None and child_b is not None:
            sep_value = self._separation_quality(child_a["cr"], child_b["cr"])
        if sep_value is None:
            sep_value = 1.0

        if quality_override is None:
            c01 = ((coherence + 1.0) * 0.5).clamp(0.0, 1.0)
            r01 = ((reliability + 1.0) * 0.5).clamp(0.0, 1.0)
            quality = (
                c01.pow(self.quality_lambda_c)
                * r01.pow(self.quality_lambda_r)
                * torch.as_tensor(sep_value, device=samples.device).clamp(0.0, 1.0).pow(self.quality_lambda_d)
            )
            quality = float(quality.detach().cpu().item())
        else:
            quality = float(quality_override)

        return {
            "samples": samples,
            "children": children,
            "parent": -1,
            "virtual": virtual,
            "cv": cv,
            "ct": ct,
            "cr": cr,
            "quality": quality,
        }

    def _cluster_reliability(self, samples, cv, ct, image_features, text_features):
        vt = (cv * ct).sum()
        i2t = (image_features[samples] @ ct).mean()
        t2i = (text_features[samples] @ cv).mean()
        return (
            self.reliability_lambda_vt * vt
            + self.reliability_lambda_i2t * i2t
            + self.reliability_lambda_t2i * t2i
        ).clamp(-1.0, 1.0)

    def _separation_quality(self, ca, cb):
        dist = 1.0 - (ca * cb).sum().clamp(-1.0, 1.0)
        left = torch.sigmoid((dist - self.sep_min) / max(self.sep_tau, EPS))
        right = torch.sigmoid((self.sep_max - dist) / max(self.sep_tau, EPS))
        return float((left * right).detach().cpu().item())

    def _initial_candidate_pairs(self, mutual_mask, sample_to_node):
        row, col = torch.nonzero(torch.triu(mutual_mask, diagonal=1), as_tuple=True)
        pairs = set()
        for i, j in zip(row.detach().cpu().tolist(), col.detach().cpu().tolist()):
            a = int(sample_to_node[i].item())
            b = int(sample_to_node[j].item())
            if a >= 0 and b >= 0 and a != b:
                if a > b:
                    a, b = b, a
                pairs.add((a, b))
        return pairs

    def _push_candidate(self, heap, nodes, a, b, affinity, image_features, text_features, fused_features):
        if a == b:
            return
        score, quality, bar_aff = self._merge_score(
            nodes[a], nodes[b], affinity, image_features, text_features, fused_features
        )
        if quality < self.min_merge_quality or bar_aff < self.min_merge_affinity or score <= 0.0:
            return
        if a > b:
            a, b = b, a
        heapq.heappush(heap, (-float(score), len(heap), a, b))

    def _merge_score(self, node_a, node_b, affinity, image_features, text_features, fused_features):
        sa = node_a["samples"]
        sb = node_b["samples"]
        if sa.numel() == 0 or sb.numel() == 0:
            return 0.0, 0.0, 0.0
        bar_aff = affinity[sa][:, sb].mean()
        samples = torch.cat([sa, sb], dim=0)
        cv = l2_normalize(image_features[samples].mean(dim=0, keepdim=True)).squeeze(0)
        ct = l2_normalize(text_features[samples].mean(dim=0, keepdim=True)).squeeze(0)
        cr = l2_normalize(fused_features[samples].mean(dim=0, keepdim=True)).squeeze(0)
        coherence = (fused_features[samples] @ cr).mean().clamp(-1.0, 1.0)
        reliability = self._cluster_reliability(samples, cv, ct, image_features, text_features)
        sep = self._separation_quality(node_a["cr"], node_b["cr"])
        quality = (
            ((coherence + 1.0) * 0.5).clamp(0.0, 1.0).pow(self.quality_lambda_c)
            * ((reliability + 1.0) * 0.5).clamp(0.0, 1.0).pow(self.quality_lambda_r)
            * torch.as_tensor(sep, device=affinity.device).clamp(0.0, 1.0).pow(self.quality_lambda_d)
        )
        score = bar_aff * quality
        return float(score.detach().cpu().item()), float(quality.detach().cpu().item()), float(bar_aff.detach().cpu().item())

    def _finalize_depth(self, nodes, root_id):
        device = nodes[0]["cr"].device
        parent = torch.tensor([n["parent"] for n in nodes], dtype=torch.long, device=device)
        depth = torch.zeros(len(nodes), dtype=torch.long, device=device)
        stack = [(root_id, 0)]
        while stack:
            node_id, d = stack.pop()
            depth[node_id] = d
            for child in nodes[node_id]["children"]:
                stack.append((child, d + 1))
        return parent, depth

    def _make_sibling_pairs(self, nodes, parent, virtual_root, device):
        by_parent = defaultdict(list)
        for child, p in enumerate(parent.detach().cpu().tolist()):
            if p < 0 or p == virtual_root:
                continue
            by_parent[p].append(child)

        pairs = []
        for children in by_parent.values():
            if len(children) < 2:
                continue
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    pairs.append((children[i], children[j]))

        if not pairs:
            return torch.empty(0, 2, dtype=torch.long, device=device)
        if len(pairs) > self.max_sibling_pairs:
            step = max(1, len(pairs) // self.max_sibling_pairs)
            pairs = pairs[::step][:self.max_sibling_pairs]
        return torch.tensor(pairs, dtype=torch.long, device=device)

    def _make_leaf_paths(self, parent, leaves):
        paths = []
        max_len = 0
        for leaf in leaves.tolist():
            path = []
            node = int(leaf)
            while node >= 0:
                path.append(node)
                node = int(parent[node].item())
            path = list(reversed(path))
            paths.append(path)
            max_len = max(max_len, len(path))

        path_tensor = torch.full((len(paths), max_len), -1, dtype=torch.long, device=parent.device)
        path_mask = torch.zeros((len(paths), max_len), dtype=torch.bool, device=parent.device)
        for i, path in enumerate(paths):
            path_tensor[i, :len(path)] = torch.tensor(path, dtype=torch.long, device=parent.device)
            path_mask[i, :len(path)] = True
        return path_tensor, path_mask

    def forward(self, noisy_image_features, noisy_text_features, hierarchy, sample_weights=None):
        if hierarchy is None or noisy_image_features.numel() == 0:
            zero = noisy_image_features.sum() * 0.0
            return {
                "loss": zero,
                "calibration_loss": zero,
                "hierarchy_loss": zero,
                "selected_depth_mean": zero.detach(),
                "valid_ratio": zero.detach(),
                "tree_confidence_mean": zero.detach(),
            }

        device = noisy_image_features.device
        dtype = noisy_image_features.dtype

        node_features = hierarchy["node_features"].to(device=device, dtype=dtype)
        leaf_features = hierarchy["leaf_features"].to(device=device, dtype=dtype)
        leaves = hierarchy["leaves"].to(device=device)
        leaf_paths = hierarchy["leaf_paths"].to(device=device)
        leaf_path_mask = hierarchy["leaf_path_mask"].to(device=device)
        parent = hierarchy["parent"].to(device=device)
        depth = hierarchy["depth"].to(device=device)
        node_quality = hierarchy["node_quality"].to(device=device, dtype=dtype)
        sibling_pairs = hierarchy["sibling_pairs"].to(device=device)
        virtual_root = int(hierarchy["virtual_root"].to(device=device).item())

        img = l2_normalize(noisy_image_features.float())
        txt = l2_normalize(noisy_text_features.float())

        if self.use_hyperbolic:
            z_img = self.image_projector(img)
            z_txt = self.text_projector(txt)
            z_node = self.node_projector(node_features.float())
            leaf_z = z_node[leaves]
            with torch.no_grad():
                scores_v = -poincare_distance(z_img[:, None, :], leaf_z[None, :, :], c=self.curvature)
                scores_t = -poincare_distance(z_txt[:, None, :], leaf_z[None, :, :], c=self.curvature)
        else:
            z_img = img
            z_txt = txt
            z_node = node_features.float()
            with torch.no_grad():
                scores_v = img @ leaf_features.float().t()
                scores_t = txt @ leaf_features.float().t()

        with torch.no_grad():
            ancestor_weights, depth_mean, valid_ratio, tree_conf_mean = self._topk_lca_weights(
                scores_v=scores_v.float(),
                scores_t=scores_t.float(),
                leaf_paths=leaf_paths,
                leaf_path_mask=leaf_path_mask,
                node_quality=node_quality.float(),
                depth=depth,
                virtual_root=virtual_root,
                sample_weights=sample_weights,
                device=device,
            )

        nz = ancestor_weights.nonzero(as_tuple=False)
        if nz.numel() == 0:
            calibration_loss = z_img.sum() * 0.0
        else:
            batch_idx = nz[:, 0]
            node_idx = nz[:, 1]
            weights = ancestor_weights[batch_idx, node_idx].to(z_img.dtype)
            if self.use_hyperbolic:
                dist_v = poincare_distance(z_img[batch_idx], z_node[node_idx], c=self.curvature)
                dist_t = poincare_distance(z_txt[batch_idx], z_node[node_idx], c=self.curvature)
                per_pair = dist_v + dist_t
            else:
                node_proto = z_node[node_idx]
                per_pair = (1.0 - (z_img[batch_idx] * node_proto).sum(dim=-1))
                per_pair = per_pair + (1.0 - (z_txt[batch_idx] * node_proto).sum(dim=-1))
            calibration_loss = (weights * per_pair).sum() / weights.sum().clamp_min(EPS)

        hierarchy_loss = self._hierarchy_loss(z_node, parent, depth, node_quality, sibling_pairs, virtual_root)
        return {
            "loss": calibration_loss + hierarchy_loss,
            "calibration_loss": calibration_loss,
            "hierarchy_loss": hierarchy_loss,
            "selected_depth_mean": depth_mean.detach(),
            "valid_ratio": valid_ratio.detach(),
            "tree_confidence_mean": tree_conf_mean.detach(),
        }

    def _topk_lca_weights(
        self,
        scores_v,
        scores_t,
        leaf_paths,
        leaf_path_mask,
        node_quality,
        depth,
        virtual_root,
        sample_weights,
        device,
    ):
        batch_size, num_leaves = scores_v.shape
        if num_leaves == 0:
            return (
                torch.zeros(batch_size, depth.numel(), device=device),
                torch.zeros((), device=device),
                torch.zeros((), device=device),
                torch.zeros((), device=device),
            )

        probs_v = F.softmax(scores_v / max(self.query_tau, EPS), dim=-1)
        probs_t = F.softmax(scores_t / max(self.query_tau, EPS), dim=-1)
        k = min(max(1, self.topk_lca), num_leaves)
        top_v_prob, top_v_pos = probs_v.topk(k=k, dim=-1)
        top_t_prob, top_t_pos = probs_t.topk(k=k, dim=-1)

        if num_leaves > 1:
            top2_v = probs_v.topk(k=2, dim=-1).values
            top2_t = probs_t.topk(k=2, dim=-1).values
            conf_v = top2_v[:, 0] - top2_v[:, 1]
            conf_t = top2_t[:, 0] - top2_t[:, 1]
        else:
            conf_v = probs_v[:, 0]
            conf_t = probs_t[:, 0]
        tree_conf = torch.minimum(conf_v, conf_t)
        conf_gate = (tree_conf > self.query_conf_threshold).float()

        if sample_weights is None:
            sample_gate = torch.ones(batch_size, device=device)
        else:
            sample_gate = sample_weights.to(device=device, dtype=torch.float32).view(-1).clamp_min(0.0)

        ancestor_probs = torch.zeros(batch_size, depth.numel(), device=device)
        for b in range(batch_size):
            if conf_gate[b].item() <= 0.0 or sample_gate[b].item() <= 0.0:
                continue
            for iv in range(k):
                pv = top_v_prob[b, iv]
                pos_v = int(top_v_pos[b, iv].item())
                for it in range(k):
                    pt = top_t_prob[b, it]
                    pos_t = int(top_t_pos[b, it].item())
                    anc = self._lca_from_leaf_positions(
                        leaf_paths, leaf_path_mask, pos_v, pos_t
                    )
                    if anc < 0 or anc == virtual_root:
                        continue
                    if int(depth[anc].item()) < self.min_ancestor_depth:
                        continue
                    q = node_quality[anc].clamp(0.0, 1.0)
                    if q.item() <= 0.0:
                        continue
                    ancestor_probs[b, anc] += pv * pt

        row_sum = ancestor_probs.sum(dim=1, keepdim=True)
        nonzero = row_sum.squeeze(1) > 0
        ancestor_probs = torch.where(
            nonzero[:, None],
            ancestor_probs / row_sum.clamp_min(EPS),
            ancestor_probs,
        )
        ancestor_weights = (
            ancestor_probs
            * node_quality.clamp(0.0, 1.0)[None, :]
            * sample_gate[:, None]
            * conf_gate[:, None]
        )

        final_row_sum = ancestor_weights.sum(dim=1)
        valid = final_row_sum > 0
        if valid.any():
            depth_float = depth.float()
            depth_per = (ancestor_weights * depth_float[None, :]).sum(dim=1) / final_row_sum.clamp_min(EPS)
            depth_mean = depth_per[valid].mean()
        else:
            depth_mean = torch.zeros((), device=device)
        valid_ratio = valid.float().mean()
        tree_conf_mean = tree_conf.mean() if tree_conf.numel() > 0 else torch.zeros((), device=device)
        return ancestor_weights, depth_mean, valid_ratio, tree_conf_mean

    def _lca_from_leaf_positions(self, leaf_paths, leaf_path_mask, pos_a, pos_b):
        path_a = leaf_paths[pos_a]
        path_b = leaf_paths[pos_b]
        mask = leaf_path_mask[pos_a] & leaf_path_mask[pos_b] & (path_a == path_b)
        if not mask.any():
            return -1
        return int(path_a[mask].long()[-1].item())

    def _hierarchy_loss(self, z_node, parent, depth, node_quality, sibling_pairs, virtual_root):
        if not self.use_hyperbolic:
            return z_node.sum() * 0.0

        child_idx = torch.arange(parent.numel(), device=z_node.device)
        valid = parent >= 0
        if virtual_root >= 0:
            valid = valid & (parent != virtual_root) & (child_idx != virtual_root)
        valid = valid & (node_quality > 0)
        if valid.sum() == 0:
            return z_node.sum() * 0.0

        child_idx = child_idx[valid]
        parent_idx = parent[valid]
        weights = node_quality[child_idx].float()
        parent_z = z_node[parent_idx]
        child_z = z_node[child_idx]

        edge_dist = poincare_distance(parent_z, child_z, c=self.curvature)
        pc = F.relu(edge_dist - self.parent_margin)
        depth_order = F.relu(parent_z.norm(dim=-1) - child_z.norm(dim=-1) + self.depth_margin)
        norm = weights.sum().clamp_min(EPS)
        parent_child_loss = (weights * pc).sum() / norm
        depth_loss = (weights * depth_order).sum() / norm

        sibling_loss = z_node.sum() * 0.0
        if sibling_pairs.numel() > 0:
            a = sibling_pairs[:, 0]
            b = sibling_pairs[:, 1]
            pair_w = torch.minimum(node_quality[a], node_quality[b]).float()
            pair_valid = pair_w > 0
            if pair_valid.any():
                a = a[pair_valid]
                b = b[pair_valid]
                pair_w = pair_w[pair_valid]
                dist = poincare_distance(z_node[a], z_node[b], c=self.curvature)
                sib = F.relu(self.sibling_margin - dist)
                sibling_loss = (pair_w * sib).sum() / pair_w.sum().clamp_min(EPS)

        return parent_child_loss + depth_loss + sibling_loss
