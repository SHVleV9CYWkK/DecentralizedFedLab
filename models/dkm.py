import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTeacherDKMLayer(nn.Module):
    """
    Differentiable k-Means with multi-teacher soft alignment
    — 每次 forward 内部 E/M；质心非 Parameter
    — 语义 (Jaccard) × 数值 (exp-dist) 混合
    """

    def __init__(
        self,
        n_clusters: int = 16,
        max_iter: int = 10,
        tau: float = 1e-3,
        eps: float = 1e-4,
        alpha_mix: float = 0.7,     # 语义占比 α
        beta_dist: float = 1.0,     # 数值距离温度 β
        beta_sem: float = 5.0,      # 教师软索引篇温度 β
    ):
        super().__init__()
        self.K = n_clusters
        self.T_max = max_iter
        self.tau = tau
        self.eps = eps
        self.alpha_mix = alpha_mix
        self.beta_dist = beta_dist
        self.beta_sem = beta_sem
        self.C = None

    def forward(
        self,
        X: torch.Tensor,                  # (N, D)
        *,
        teacher_centroids: torch.Tensor | None = None,   # (T, K, D)
        teacher_alphas: torch.Tensor | None = None,      # (T,)
        teacher_index_tables: list[torch.Tensor] | None = None,  # len T, (N_t,)
        lambda_teacher: float = 1.0,
    ):
        """
        返回:
            X_rec : (N,D)   重构特征
            C     : (K,D)   学生质心
            A     : (N,K)   soft assignment
        """
        X = X - X.mean(dim=0)                       # 中心化
        device = X.device

        # 初始化质心 (k-means++)
        if self.C is None:
            C = self._kpp_init(X)                       # (K,D)
        else:
            C = self.C

        # 教师相关预处理
        use_teacher = teacher_centroids is not None and lambda_teacher > 0.0
        if use_teacher:
            T = teacher_centroids.size(0)
            teacher_centroids = teacher_centroids.to(device)
            if teacher_alphas is None:
                teacher_alphas = torch.ones(T, device=device) / T
            else:
                teacher_alphas = teacher_alphas.to(device)

            if teacher_index_tables is not None:
                # one-hot -> (T, K, N_t)
                teacher_oh = [
                    F.one_hot(lbl.to(device), num_classes=self.K).float().t()
                    for lbl in teacher_index_tables
                ]
        else:
            lambda_teacher = 0.0

        # E/M 迭代
        for _ in range(self.T_max):
            # E-step 估计隐变量（软分配概率）
            dist2 = torch.cdist(X, C, p=2).pow(2)          # (N,K)
            A = F.softmax(-dist2 / self.tau, dim=1)        # (N,K)

            # 教师软对齐
            if use_teacher:
                if teacher_index_tables is None:
                    # teacher_soft[t] : (N,K)
                    teacher_soft = []
                    for t in range(T):
                        dist_xc = torch.cdist(X, teacher_centroids[t], p=2)  # (N,K)
                        prob = torch.softmax(-self.beta_sem * dist_xc, dim=1)
                        teacher_soft.append(prob)

                    # 计算 soft-Jaccard
                    jac_list = []
                    for t in range(T):
                        inter = teacher_soft[t].T @ A  # (K,K)
                        union = (
                                teacher_soft[t].sum(0, keepdim=True).T +
                                A.sum(0, keepdim=True) - inter + 1e-8
                        )
                        jac_list.append(inter / union)
                    J = torch.stack(jac_list, 0)  # (T,K,K)
                else:
                    # 1) Jaccard 语义相似  (T,K,K)
                    #   先取学生硬标签以加速；对梯度影响可忽略
                    stu_labels = A.argmax(dim=1)                       # (N,)
                    stu_oh = F.one_hot(stu_labels, num_classes=self.K).float().t()  # (K,N)

                    jac_list = []
                    for t in range(T):
                        inter = teacher_oh[t] @ stu_oh.t()             # (K,K)
                        union = (
                            teacher_oh[t].sum(dim=1, keepdim=True)
                            + stu_oh.sum(dim=1, keepdim=True) - inter + 1e-8
                        )
                        jac_list.append(inter / union)
                    J = torch.stack(jac_list, dim=0)                   # (T,K,K)

                # 2) 数值距离相似 S
                dist_c = torch.cdist(teacher_centroids, C, p=2)        # (T,K,K)
                S = torch.exp(-self.beta_dist * dist_c)

                # 3) 混合权重 M -> w 行归一化 幂次混合
                # 只有教师质心都很契合学生时，才给予学生大的引导权重
                M = (J + 1e-8).pow(self.alpha_mix) * \
                    (S + 1e-8).pow(1.0 - self.alpha_mix)
                w = M / (M.sum(dim=2, keepdim=True) + 1e-8)            # (T,K,K) 行归一化，得到对齐权重

                # 4) teacher_matched_j  = Σ_{t,i} α_t * w_{t,i,j} * C^{t}_i
                teacher_matched = torch.einsum(
                    't, tik, t i d -> k d', teacher_alphas, w, teacher_centroids
                )                                                      # (K,D)
            else:
                teacher_matched = 0.0

            # M-step
            num_stu = A.sum(dim=0, keepdim=True).t()                   # (K,1)
            C_new = (A.t() @ X + lambda_teacher * teacher_matched) / (num_stu + lambda_teacher + 1e-8)

            if torch.norm(C_new - C) < self.eps:
                C = C_new
                break
            C = C_new

        # 重构
        X_rec = A @ C                                                 # (N,D)
        self.C = C
        return X_rec, C, A

    def _kpp_init(self, X):
        """k-means++ 初始化, 返回 (K,D)"""
        N, _ = X.shape
        device = X.device
        idx = torch.randint(0, N, (1,), device=device)
        C = [X[idx].squeeze(0)]
        for _ in range(1, self.K):
            D2 = torch.cdist(X, torch.stack(C), p=2).pow(2).min(dim=1)[0]
            probs = D2 / (D2.sum() + 1e-8)
            nxt = torch.multinomial(probs, 1)
            C.append(X[nxt].squeeze(0))
        return torch.stack(C)
