import torch
import torch.nn as nn
import torch.nn.functional as F

from clients.dfl_method_clients.dfedmtkd_client import DFedMTKDClient   # ← 已优化的父类


# ------------------------ 工具函数 ------------------------
def _samplewise_mse(a, b):
    return F.mse_loss(a, b, reduction="none").view(a.size(0), -1).mean(dim=1)


# ------------------------ Agent MLP ------------------------
class AgentMLP(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)           # [N] → [N]


# ======================== 主客户端 =========================
class DFedMTKDRLClient(DFedMTKDClient):
    """
    在 DFedMTKD 基础上加入“强化学习动态权重”：
    * 内存友好：不再保留 teacher 全数据 logits/feature，而是 batch-wise 推理
    * 只保留一份 teacher_model & hook
    * agent 只接收当前 batch 的 state，梯度正常回传
    """
    # ---------- 1. 初始化 ----------
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        self.lambda_logit_kd   = hyperparam.get("lambda_kd", 0.1)
        self.lambda_feature_kd = hyperparam.get("lambda_feature_kd", 0.1)

        self.state_dim   = hyperparam.get("agent_state_dim", 4)
        self.hidden_dim  = hyperparam.get("agent_hidden_dim", 64)
        self.agent_lr    = hyperparam.get("agent_lr", 1e-3)

        # 训练阶段标志位
        self._student_pretrained = False
        self._agent_pretrained   = False

        # Agent 及优化器
        self.agent            = None
        self.agent_optimizer  = None

        # 复用 0 标量，避免每批新建
        self.zero_scalar      = torch.zeros((), device=self.device)

    # ---------- 2. 初始化本地数据加载器 / agent ----------
    def init_client(self):
        super().init_client()                                   # 父类已创建 model / optimizer
        self.agent           = AgentMLP(self.state_dim, self.hidden_dim).to(self.device)
        self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.agent_lr)

    # ---------- 3. teacher 即时推理 ----------
    def _forward_teacher_batch(self, batch_x):
        """
        给定一批 x，逐个 neighbor teacher 前向，返回：
            - tea_logits_list : List[Tensor[B, C]]
            - tea_feats_list  : List[Tensor[B, ...]]
        其中 feats 来源于 teacher hook 捕获。
        """
        tea_logits_list, tea_feats_list = [], []
        if not self.neighbor_model_weights:
            return tea_logits_list, tea_feats_list

        with torch.inference_mode():
            for t_state in self.neighbor_model_weights:         # M 个 teacher
                # 就地参数覆盖（父类已实现 _student_hook & _teacher_hook）
                self._load_teacher_state_inplace(t_state)

                self.teacher_features_outputs.clear()
                logits = self.teacher_model(batch_x)
                feat   = self.teacher_features_outputs[-1]      # 最后一次 hook
                self.teacher_features_outputs.clear()

                tea_logits_list.append(logits)
                tea_feats_list .append(feat)

        return tea_logits_list, tea_feats_list                  # List 长度 = M

    # ---------- 工具：就地更新 teacher_model ----------
    def _load_teacher_state_inplace(self, state_dict):
        tgt_params  = dict(self.teacher_model.named_parameters())
        tgt_buffers = dict(self.teacher_model.named_buffers())
        for name, src in state_dict.items():
            if name in tgt_params:
                tgt_params[name].copy_(src, non_blocking=True)
            elif name in tgt_buffers:
                tgt_buffers[name].copy_(src, non_blocking=True)

    # ---------- 4. 构造 agent state ----------
    def _build_state(self, labels, stu_logits, stu_feat, tea_logits_list, tea_feat_list):
        """
        按 **当前 batch** 构造状态张量 [B, M, 4]：
            [cross-entropy, cosine, KL, teacher confidence]
        """
        B = labels.size(0)
        M = len(tea_logits_list)
        stu_feat_flat = stu_feat.view(B, -1)

        state_elems = []
        for tea_logits, tea_feat in zip(tea_logits_list, tea_feat_list):
            tea_feat_flat = tea_feat.view(B, -1)

            ce   = F.cross_entropy(tea_logits, labels, reduction="none")            # [B]
            cos  = F.cosine_similarity(stu_feat_flat, tea_feat_flat, dim=-1)        # [B]
            kl   = F.kl_div(
                      F.log_softmax(stu_logits, dim=1),
                      F.softmax(tea_logits, dim=1),
                      reduction="none"
                  ).sum(-1)                                                         # [B]
            conf = tea_logits.softmax(-1).mean(-1)                                  # [B]

            state_elems.append(torch.stack([ce, cos, kl, conf], dim=-1))            # [B, 4]

        return torch.stack(state_elems, dim=1)                                      # [B, M, 4]

    # ---------- 5. Agent 推理 ----------
    def _run_agent(self, state):
        B, M, D = state.shape
        logits  = self.agent(state.view(B * M, D)).view(B, M)  # [B, M]
        return torch.softmax(logits, dim=1)                    # 权重和为 1

    # ---------- 6. 预训练阶段 ----------
    def _pretrain_student(self):
        """用 teacher 硬 KD 先训学生 1 个 epoch（可选）"""
        if not self.neighbor_model_weights:
            return False
        super().train()                                        # 父类基于即时 KD 的 train()
        return True

    def _pretrain_agent(self):
        """用均匀权重作为目标，预训 agent"""
        if not self.neighbor_model_weights:
            return False

        M      = len(self.neighbor_model_weights)
        target = torch.full((1, M), 1 / M, device=self.device)  # [1, M]
        self.agent.train()

        for x, labels in self.client_train_loader:
            x, labels = x.to(self.device), labels.to(self.device)

            # --- 学生前向（冻结梯度） ---
            self.features_buffer.clear()
            stu_logits = self.model(x)
            stu_feat   = self.features_buffer[-1]

            tea_logits_list, tea_feat_list = self._forward_teacher_batch(x)

            state = self._build_state(labels, stu_logits, stu_feat,
                                        tea_logits_list, tea_feat_list)            # [B, M, D]
            weights = self._run_agent(state.detach())                                        # [B, M]
            loss = F.mse_loss(weights, target.expand_as(weights))

            self.agent_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.agent_optimizer.step()

        return True

    # ---------- 7. 主训练 ----------
    def train(self):
        # -- 7.1 可选预训练阶段 --
        if not self._student_pretrained:
            self._student_pretrained = self._pretrain_student()

        if not self._agent_pretrained:
            self._agent_pretrained = self._pretrain_agent()

        # -- 7.2 正式联合训练 --
        for epoch in range(self.epochs):
            self.model.train()
            buffer_logprob, buffer_reward = [], []

            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                self.features_buffer.clear()

                # ---- 学生前向 ----
                stu_logits = self.model(x)

                # ---- Teacher forward + 状态 ----
                loss_logit_kd   = self.zero_scalar.clone()
                loss_feature_kd = self.zero_scalar.clone()

                tea_logits_list, tea_feat_list = [], []
                weights, log_prob = None, None

                if self.neighbor_model_weights:
                    # 即时推理所有 teacher
                    tea_logits_list, tea_feat_list = self._forward_teacher_batch(x)

                    stu_feat = self.features_buffer[-1]

                    if self._agent_pretrained:                 # RL agent 已可用
                        state = self._build_state(labels, stu_logits, stu_feat,
                                                    tea_logits_list, tea_feat_list)
                        weights = self._run_agent(state.detach())       # [B, M]
                        log_prob = torch.log(weights + 1e-8)   # 留给 policy gradient

                    # --- KD 损失 ---
                    for m, (tea_logits, tea_feat) in enumerate(zip(tea_logits_list, tea_feat_list)):
                        # 权重；若未预训 agent，则均匀
                        w = 1.0 / len(tea_logits_list) if weights is None else weights[:, m].detach()

                        # KL
                        kl = F.kl_div(
                                F.log_softmax(stu_logits, dim=1),
                                F.softmax(tea_logits, dim=1),
                                reduction="none"
                             ).sum(-1)                          # [B]
                        # MSE
                        mse = _samplewise_mse(stu_feat, tea_feat)  # [B]

                        loss_logit_kd   += (w * kl).mean()
                        loss_feature_kd += (w * mse).mean()

                # ---- 总 loss & 反向 ----
                loss_sup = self.criterion(stu_logits, labels).mean()
                loss     = (loss_sup +
                            self.lambda_logit_kd   * loss_logit_kd +
                            self.lambda_feature_kd * loss_feature_kd)

                loss.backward()
                self.optimizer.step()

                # ---- 累积 RL buffer (只在 agent 阶段) ----
                if self._agent_pretrained and self.neighbor_model_weights:
                    # reward 依据教师损失越小越好，取负
                    kl_det  = torch.stack([
                                  F.kl_div(
                                      F.log_softmax(stu_logits, 1),
                                      F.softmax(tl, 1),
                                      reduction="none"
                                  ).sum(-1) for tl in tea_logits_list
                              ], dim=1)                    # [B, M]

                    mse_det = torch.stack([
                                  _samplewise_mse(stu_feat, tf)
                              for tf in tea_feat_list], dim=1)  # [B, M]

                    ce_det  = F.cross_entropy(stu_logits, labels, reduction='none')  # [B]
                    reward  = - (ce_det.unsqueeze(1) +                              # [B, M]
                                 self.lambda_logit_kd   * kl_det +
                                 self.lambda_feature_kd * mse_det)

                    buffer_logprob.append(log_prob)   # [B, M]，带梯度
                    buffer_reward .append(reward)     # [B, M]，无梯度

            # ---- 7.3 更新 RL agent ----
            if self._agent_pretrained and buffer_logprob:
                log_prob = torch.cat(buffer_logprob, dim=0)      # [T, M]
                reward   = torch.cat(buffer_reward,  dim=0)      # [T, M]

                # 奖励归一化
                R_min   = reward.min(dim=1, keepdim=True).values
                R_max   = reward.max(dim=1, keepdim=True).values
                R_norm  = (reward - R_min) / (R_max - R_min + 1e-8)
                R_mean  = reward.mean(dim=1, keepdim=True)
                R_final = R_norm - R_mean

                policy_loss = -(R_final.detach() * log_prob).mean()
                self.agent_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                self.agent_optimizer.step()

        self.neighbor_model_weights.clear()