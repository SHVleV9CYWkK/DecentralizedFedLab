from collections import defaultdict, deque
import torch

from clients.dfl_method_clients.dfedcad_client import DFedCADClient, _cfd_distance
from utils.kmeans import TorchKMeans


# Decentralized Federated Maturity-Aligned Centroids (with transmitted maturity & async rounds)
class DFedMACClient(DFedCADClient):
    """
    DFedCAD + 质心成熟度（C-SWAG 置信度 × 稳定度）+ 参数侧柔性加权对齐
    - 不使用 ref_momentum，仅靠结构对齐贡献知识
    - 发送“自己的成熟度”（层级标量），接收端优先使用对端成熟度
    - 维护本地通信轮数 self.local_round（完全异步）
    """
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        # ==== 成熟度相关超参 ====
        self.maturity_window = hyperparam.get('maturity_window', 5)  # C-SWAG窗口长度
        self.beta_drift = hyperparam.get('beta_drift', 1.0)          # 稳定度：漂移项系数
        self.beta_invar = hyperparam.get('beta_invar', 1.0)          # 稳定度：簇内方差项系数
        self.beta_mask = hyperparam.get('beta_mask', 0.5)            # 稳定度：mask翻转率项系数
        self.maturity_eps = hyperparam.get('maturity_eps', 1e-8)

        # ==== 柔性降权（替代 Top-p 与 剔除阈值） ====
        # 簇层面：w_k = clip( (maturity_k / mean_maturity) ** cluster_gamma, [cluster_floor, cluster_cap] )
        self.cluster_gamma = hyperparam.get('cluster_gamma', 1.0)
        self.cluster_floor = hyperparam.get('cluster_floor', 0.2)
        self.cluster_cap   = hyperparam.get('cluster_cap', 2.0)

        # 教师层面：alpha_eff = blend( base_alpha , normalize(base_alpha * maturity^teacher_gamma), teacher_blend )
        self.teacher_gamma = hyperparam.get('teacher_gamma', 1.0)
        self.teacher_blend = hyperparam.get('teacher_blend', 0.8)  # 1.0 更依赖成熟度，0.0 更依赖基础相似度

        # 当对端也发来成熟度时，与本地估计的融合比例（这里优先信对端发来的层级标量）
        self.teacher_tx_blend = hyperparam.get('teacher_tx_blend', 0.8)  # 0~1

        # 历史缓存（用于本地成熟度）：分层记录质心 & mask
        self._local_hist = defaultdict(lambda: deque(maxlen=self.maturity_window))  # {layer: [K×1 质心]}
        self._local_mask_hist = defaultdict(lambda: deque(maxlen=2))               # {layer: [mask_bool]}

        # 教师历史（可选备用；邻接若频繁变化，此缓存用处有限，但保留以兼容无成熟度发送的邻居）
        self._teacher_hist = []  # list[ defaultdict(layer -> deque of centroids) ]
        self._teacher_ready = False

        # ==== 异步轮数 ====
        # 记录本地完成“训练→聚类→发送”一次的通信轮数
        self.local_round = 0

    # ---------- 小工具：按值排序质心并重映射标签 ----------
    @staticmethod
    def _sort_centroids_and_remap(centroids_1d, labels_1d):
        """
        centroids_1d: Tensor [K] 或 [K,1]
        labels_1d:    LongTensor [N]
        返回：sorted_centroids [K,1], remapped_labels [N]
        """
        c = centroids_1d.view(-1).detach()
        order = torch.argsort(c)  # 升序
        sorted_c = c[order].view(-1, 1)
        # 旧 -> 新 的映射
        old2new = torch.empty_like(order)
        old2new[order] = torch.arange(order.numel(), device=order.device)
        new_labels = old2new[labels_1d]
        return sorted_c, new_labels

    # ---------- 统计：每簇簇内方差 ----------
    @staticmethod
    def _cluster_intra_var(flat_weights, labels, K, eps=1e-8):
        """
        flat_weights: Tensor [N,1]
        labels:       LongTensor [N]
        返回：每簇方差 Tensor [K,1]
        """
        vars_out = flat_weights.new_zeros(K, 1)
        for k in range(K):
            idx = (labels == k)
            if idx.any():
                w = flat_weights[idx].view(-1)
                vars_out[k, 0] = torch.var(w, unbiased=False)
            else:
                vars_out[k, 0] = eps
        return vars_out + eps

    # ---------- 本地历史更新（用于 C-SWAG & 稳定度） ----------
    def _update_local_history(self, centroids_dict, mask_dict):
        for layer, c in centroids_dict.items():
            self._local_hist[layer].append(c.detach().cpu())
        for layer, m in mask_dict.items():
            self._local_mask_hist[layer].append(m.detach().cpu())

    # ---------- 教师历史结构确保 & 更新（备用） ----------
    def _ensure_teacher_hist(self, num_teachers):
        if not self._teacher_ready or len(self._teacher_hist) != num_teachers:
            self._teacher_hist = [
                defaultdict(lambda: deque(maxlen=self.maturity_window)) for _ in range(num_teachers)
            ]
            self._teacher_ready = True

    def _update_teacher_history(self, teacher_idx, teacher_centroids_dict):
        if not self._teacher_ready:
            return
        for layer, c in teacher_centroids_dict.items():
            self._teacher_hist[teacher_idx][layer].append(c.detach().cpu())

    # ---------- C-SWAG 置信度（长期）：质心快照方差的倒数 ----------
    def _c_swag_precision(self, hist_deque):
        """
        hist_deque: deque of [K,1] tensors
        返回：lambda Tensor [K,1]
        """
        if len(hist_deque) < 2:
            k = hist_deque[-1].shape[0]
            return hist_deque[-1].new_ones(k, 1)
        stack = torch.stack(list(hist_deque), dim=0)  # [T,K,1]
        var = torch.var(stack, dim=0, unbiased=False) # [K,1]
        return 1.0 / (var + self.maturity_eps)

    # ---------- 稳定度（短期）：漂移 + 簇内方差 + mask翻转 ----------
    def _stability_scores(self, layer_key, centroids_now, labels_now, mask_now):
        """
        返回：stab Tensor [K,1]
        """
        device = centroids_now.device
        # 漂移：和上一帧质心差
        if len(self._local_hist[layer_key]) >= 1:
            prev_c = self._local_hist[layer_key][-1].to(device)
            drift = torch.abs(centroids_now - prev_c)  # [K,1]
        else:
            drift = torch.zeros_like(centroids_now)

        # 簇内方差（当前）
        flat_w = self.model.state_dict()[layer_key].to(device).view(-1, 1).detach()
        K = centroids_now.shape[0]
        invar = self._cluster_intra_var(flat_w, labels_now, K, eps=self.maturity_eps)  # [K,1]

        # mask 翻转率（层级标量）
        if len(self._local_mask_hist[layer_key]) >= 1:
            prev_m = self._local_mask_hist[layer_key][-1].to(device)
            flips = (prev_m ^ mask_now).float().mean()  # 比例
        else:
            flips = torch.tensor(0.0, device=device)

        # 稳定度（逐簇）：exp( -β1·|Δc| - β2·σ_intra^2 - β3·flip_rate )
        stab = torch.exp(
            - self.beta_drift * drift
            - self.beta_invar * invar
            - self.beta_mask * flips
        )
        return stab.clamp_min(1e-6)

    # ---------- 本地成熟度（用于柔性簇权重） ----------
    def _local_maturity(self, layer_key, centroids_now, labels_now, mask_now):
        lam = self._c_swag_precision(self._local_hist[layer_key]).to(centroids_now.device)  # [K,1]
        stab = self._stability_scores(layer_key, centroids_now, labels_now, mask_now)       # [K,1]
        maturity = lam * stab                                                                # [K,1]
        return maturity

    # ---------- 覆盖：聚类 + 剪枝 + 返回质心/标签（修复空 dict） ----------
    def _cluster_and_prune_model_weights(self):
        clustered_state_dict = {}
        mask_dict = {}
        centroids_dict = {}
        labels_dict = {}

        for key, weight in self.model.state_dict().items():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key:
                original_shape = weight.shape
                kmeans = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)
                flat = weight.detach().view(-1, 1)
                kmeans.fit(flat)  # kmeans.centroids [K,1], kmeans.labels_ [N]

                # 质心排序 + 标签重映射（提升跨轮配对稳定性）
                centroids_sorted, labels_sorted = self._sort_centroids_and_remap(
                    kmeans.centroids.view(-1, 1), kmeans.labels_
                )

                # 重构 + mask（0 质心视为剪枝）
                new_weights = centroids_sorted[labels_sorted].view(original_shape)
                is_zero = (centroids_sorted.view(-1) == 0)
                mask = (is_zero[labels_sorted].view(original_shape) == 0)

                clustered_state_dict[key] = new_weights
                mask_dict[key] = mask.bool()
                centroids_dict[key] = centroids_sorted  # [K,1]
                labels_dict[key] = labels_sorted.view(-1)  # [N]
            else:
                clustered_state_dict[key] = weight
                mask_dict[key] = torch.ones_like(weight, dtype=torch.bool)

        # 更新本地历史（用于成熟度计算）
        self._update_local_history(centroids_dict, mask_dict)
        self.mask = mask_dict
        return clustered_state_dict, centroids_dict, labels_dict

    # ---------- 生成“层级成熟度”元信息（发送用） ----------
    def _prepare_maturity_meta(self):
        """
        生成极简成熟度元信息：
          meta = {
            'version': 'dfedmac_meta_v1',
            'client_id': self.client_id,
            'round': self.local_round,
            'layer_maturity': {layer_key: float(mean(lambda*stab))}
          }
        依赖：最近一次 _cluster_and_prune_model_weights() 已更新 _local_hist/_local_mask_hist
        """
        layer_maturity = {}
        # 确保有最新聚类结果
        _, centroids_dict, labels_dict = self._cluster_and_prune_model_weights()
        for layer_key, centroids_now in centroids_dict.items():
            device = self.device
            centroids_now = centroids_now.to(device)
            labels_now = labels_dict[layer_key].to(device)
            mask_now = self.mask[layer_key].to(device)
            maturity_vec = self._local_maturity(layer_key, centroids_now, labels_now, mask_now)  # [K,1]
            # 层级标量（简单均值；如需可改为按簇大小加权均值）
            layer_maturity[layer_key] = float(maturity_vec.mean().item())

        meta = {
            'version': 'dfedmac_meta_v1',
            'client_id': getattr(self, 'client_id', None),
            'round': int(self.local_round),
            'layer_maturity': layer_maturity
        }
        return meta

    # ---------- 教师信息（使用对端发送的层级成熟度 + CFD 相似度） ----------
    def _all_teacher_info(self):
        """
        兼容两种邻居格式：
        - 3元组：(weights, centroids_dict, labels_dict)
        - 4元组：(weights, centroids_dict, labels_dict, meta) 其中 meta['layer_maturity'] 可选
        """
        # 1) 本地质心
        _, local_centroids_dict, _ = self._cluster_and_prune_model_weights()

        # 2) 收集教师质心与可选 meta，并计算 CFD
        cfd_matrix = []
        teacher_centroids_dicts = []
        teacher_meta_list = []

        for item in self.neighbor_model_weights:
            if len(item) == 4:
                _, teacher_centroids, _, meta = item
                teacher_meta_list.append(meta)
            else:
                _, teacher_centroids, _ = item
                teacher_meta_list.append(None)
            teacher_centroids_dicts.append(teacher_centroids)

            per_layer = []
            for layer_key in local_centroids_dict:
                cfd = _cfd_distance(
                    local_centroids_dict[layer_key].detach().float(),
                    teacher_centroids[layer_key].detach().float()
                )
                per_layer.append(cfd)
            cfd_matrix.append(per_layer)

        if len(cfd_matrix) == 0:
            self.teacher_info_list = []
            return

        cfd_tensor = torch.tensor(cfd_matrix, dtype=torch.float, device=self.device)  # [T, L]
        cfd_scores = torch.mean(cfd_tensor, dim=1)  # [T]

        # 3) 基础相似度权重：base alpha（相似度越高权越大）
        min_val, max_val = cfd_scores.min(), cfd_scores.max()
        normed = (cfd_scores - min_val) / (max_val - min_val + 1e-8)
        beta = 2.0
        base_alphas = torch.softmax(-beta * normed, dim=0)  # [T]

        # 4) 读取并归一老师的层级成熟度（按“同层、同轮”做 min-max 归一）
        T = len(teacher_centroids_dicts)
        layer_keys = list(local_centroids_dict.keys())

        # 先组织一个 [L, T] 的成熟度表，缺失的填 1.0
        maturity_mat = torch.ones(len(layer_keys), T, dtype=torch.float, device=self.device)
        for t_idx, meta in enumerate(teacher_meta_list):
            if isinstance(meta, dict) and ('layer_maturity' in meta):
                lm = meta['layer_maturity']
                for li, layer_key in enumerate(layer_keys):
                    if layer_key in lm:
                        maturity_mat[li, t_idx] = max(float(lm[layer_key]), self.maturity_eps)

        # 对每一层做 min-max 归一到 [eps, 1]
        vmin = maturity_mat.min(dim=1, keepdim=True).values
        vmax = maturity_mat.max(dim=1, keepdim=True).values
        maturity_norm = (maturity_mat - vmin) / (vmax - vmin + 1e-8)
        maturity_norm = torch.clamp(maturity_norm, self.maturity_eps, 1.0)  # [L, T]

        # 5) 存储教师信息：基础 alpha + 每层成熟度（已归一）
        self.teacher_info_list = []
        for t in range(T):
            layer_maturity_dict = {layer_keys[li]: float(maturity_norm[li, t].item()) for li in range(len(layer_keys))}
            self.teacher_info_list.append({
                'centroids': teacher_centroids_dicts[t],
                'alpha': float(base_alphas[t].item()),
                'layer_maturity': layer_maturity_dict
            })

        # （可选）保留教师质心历史缓存，兼容没有发送成熟度的邻居
        self._ensure_teacher_hist(T)
        for t_idx in range(T):
            self._update_teacher_history(t_idx, teacher_centroids_dicts[t_idx])

    # ---------- 对齐损失（柔性簇权重 + 教师柔性降权） ----------
    def _compute_alignment_loss(self):
        if len(self.teacher_info_list) == 0:
            return torch.zeros((), device=self.device)

        losses = []
        state = self.model.state_dict()

        for layer_key, dkm in self.dkm_layers.items():
            # 学生当前权重/展平
            W = state[layer_key].to(self.device)
            Wf = W.view(-1, 1)

            # 本地聚类（为成熟度与簇权重准备）
            _, centroids_dict, labels_dict = self._cluster_and_prune_model_weights()
            centroids_now = centroids_dict[layer_key].to(self.device)              # [K,1]
            labels_now = labels_dict[layer_key].to(self.device).view(-1)           # [N]
            mask_now = self.mask[layer_key].to(self.device)

            # 本地成熟度（逐簇）
            maturity_vec = self._local_maturity(layer_key, centroids_now, labels_now, mask_now)  # [K,1]

            # —— 簇层面的柔性降权 —— #
            rel = maturity_vec / (maturity_vec.mean() + self.maturity_eps)  # [K,1]
            w_cluster = torch.clamp(rel ** self.cluster_gamma,
                                    self.cluster_floor, self.cluster_cap)   # [K,1]
            w_weight = w_cluster.view(-1)[labels_now]                       # [N]
            w_weight = w_weight / (w_weight.mean() + self.maturity_eps)     # mean≈1

            # 教师输入
            T = len(self.teacher_info_list)
            teacher_centroids = torch.stack(
                [t["centroids"][layer_key].to(self.device) for t in self.teacher_info_list], dim=0
            )  # [T,K,1]

            # —— 教师层面的柔性降权（使用对端成熟度） —— #
            # 取出该层的成熟度（已在 _all_teacher_info 中做过 min-max 归一）
            maturity_per_teacher = torch.tensor(
                [t["layer_maturity"].get(layer_key, 1.0) for t in self.teacher_info_list],
                device=self.device, dtype=torch.float
            )  # [T]

            alpha_base = torch.tensor([t["alpha"] for t in self.teacher_info_list],
                                      device=self.device, dtype=torch.float)  # [T]
            alpha_base_norm = alpha_base / (alpha_base.sum() + 1e-12)

            alpha_maturity = maturity_per_teacher ** self.teacher_gamma
            alpha_eff_raw = alpha_base * alpha_maturity
            if alpha_eff_raw.sum() <= 1e-12:
                alpha_eff_norm = torch.ones_like(alpha_eff_raw) / max(1, T)
            else:
                alpha_eff_norm = alpha_eff_raw / alpha_eff_raw.sum()

            # 平滑混合（温和使用成熟度）
            alpha_eff = (1.0 - self.teacher_blend) * alpha_base_norm + self.teacher_blend * alpha_eff_norm
            alpha_eff = alpha_eff / (alpha_eff.sum() + 1e-12)

            # DKM 重构（多教师）
            X_rec, _, _ = dkm(
                Wf,
                teacher_centroids=teacher_centroids,    # [T,K,1]
                teacher_alphas=alpha_eff,               # [T]
                teacher_index_tables=None,
                lambda_teacher=self.lambda_alignment
            )

            # 加权MSE（逐元素），再求均值
            per_elem = (Wf - X_rec).pow(2).view(-1)
            losses.append((w_weight * per_elem).mean())

        return torch.stack(losses).sum() if losses else torch.zeros((), device=self.device)

    # ---------- 本地训练（移除 ref_momentum） ----------
    def _local_train(self):
        self.model.train()
        for _, (x, labels) in enumerate(self.client_train_loader):
            # 应用剪枝 mask（保持结构一致）
            self.model.load_state_dict(self._prune_model_weights())

            x, labels = x.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(x)
            loss_sup = self.criterion(outputs, labels).mean()

            loss_align = self._compute_alignment_loss() if self.is_align else torch.zeros((), device=self.device)
            loss_final = loss_sup + self.lambda_alignment * loss_align
            loss_final.backward()
            self.optimizer.step()

    # ---------- 训练流程（完全异步：先收老师，再本地训练；结束后轮数+1） ----------
    def train(self):
        # 对齐才需要 DKM
        if self.is_align and len(self.dkm_layers) == 0:
            self._register_dkm_layers()

        # 先统计教师信息（含成熟度 & CFD）
        if self.is_align:
            self._all_teacher_info()

        # 本地训练若干 epoch
        for _ in range(self.epochs):
            self._local_train()

        # 训练结束后，准备可通信的聚类表示
        self.cluster_model = self._cluster_and_prune_model_weights()

        # 完成本地一次“训练→聚类→发送”的通信轮次
        self.local_round += 1

        # 清空邻居缓存（由上层重新注入下一批邻居）
        self.neighbor_model_weights.clear()

    # ---------- 发送模型（携带自己的层级成熟度 + 本地轮数） ----------
    def send_model(self):
        """
        返回 4 元组（向后兼容接收端）：
          (clustered_state_dict, centroids_dict, labels_dict, meta)
        其中 meta:
          - version: 'dfedmac_meta_v1'
          - client_id: 发送方ID（可选）
          - round: 发送方本地轮数（异步记录，不参与时间衰减）
          - layer_maturity: {layer_key: float 标量成熟度}
        """
        if self.cluster_model is None:
            self.cluster_model = self._cluster_and_prune_model_weights()
        clustered_state_dict, centroids_dict, labels_dict = self.cluster_model
        maturity_meta = self._prepare_maturity_meta()
        return (clustered_state_dict, centroids_dict, labels_dict, maturity_meta)