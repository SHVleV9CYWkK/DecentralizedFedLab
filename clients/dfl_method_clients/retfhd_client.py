# -*- coding: utf-8 -*-
"""clients/retfhd.py – ReT‑FHD client (multi‑level KD version)

This module upgrades the previous single‑logit implementation to **true multi‑
level knowledge‑distillation**:

* Regular clients → unchanged (FedAvg weight aggregation + local SGD)
* Delayed clients → use neighbours’ averaged **weights** as a *frozen* teacher
  and perform **multi‑level KD** with
  - **Elastic layer‑wise temperature τₗ**  (simple linear scaling demo)
  - **Category‑aware temperature τ̃_c**

> ⚠️ 真实科研场景可根据具体网络自定义 `_forward_with_intermediate()`
> 以返回想要蒸馏的各层特征。这里给出一个对常见 CNN & ResNet 结构
> 通用但不完美的实现：捕获 Conv/ReLU/FC 的 block 末输出。
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from clients.client import Client

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _log_softmax_tau(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return F.log_softmax(logits / tau, dim=-1)


def _softmax_tau(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return F.softmax(logits / tau, dim=-1)


def _kl_divergence(student_logits: torch.Tensor,
                   teacher_logits: torch.Tensor,
                   tau: float) -> torch.Tensor:
    """KL( teacher ‖ student ) with temperature scaling."""
    log_ps = _log_softmax_tau(student_logits, tau)  # log‑prob student
    with torch.no_grad():
        pt = _softmax_tau(teacher_logits, tau)      # prob teacher
    return F.kl_div(log_ps, pt, reduction="batchmean") * (tau ** 2)


# -----------------------------------------------------------------------------
# ReT‑FHD Client
# -----------------------------------------------------------------------------

class ReTFHDClient(Client):
    """Client with FedAvg (regular) **or** ReT‑FHD KD (delayed)."""

    # ──────────────────────────────── init ────────────────────────────────
    def __init__(self,
                 client_id: int,
                 dataset_index,
                 full_dataset,
                 hyperparam,
                 device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        self.is_delayed: bool = False          # role flag
        self._teacher_model: Optional[nn.Module] = None  # frozen teacher weights

        # category‑wise temperature τ̃_c  (init=1)
        self._cat_tau = torch.ones(self.num_classes, device=device)

        # base temperature hyper‑params
        self._base_tau: float = hyperparam.get("tau", 2.0)       # default layer‑base τ
        self._elastic_gamma: float = hyperparam.get("gamma", 1.0)  # scaling for τₗ

        # placeholder for hook handles (optional, not used in this simple impl.)
        self._hook_handles = []

    # ─────────────────────────── client lifecycle hooks ───────────────────────────
    def init_client(self):
        super().init_client()
        # If neighbour weights已经预缓冲 ⇒ 当前节点为 *delayed* (will use KD)
        if len(self.neighbor_model_weights) != 0:
            self.is_delayed = True
            self.aggregate()

    def aggregate(self):
        if not self.neighbor_model_weights:
            return
        aggregated_state = self._weight_aggregation()

        if self.is_delayed:
            # build/update frozen teacher
            if self._teacher_model is None:
                self._teacher_model = deepcopy(self.model).to(self.device)
                for p in self._teacher_model.parameters():
                    p.requires_grad_(False)
            self._teacher_model.load_state_dict(aggregated_state, strict=True)
        else:
            # FedAvg path
            self.model.load_state_dict(aggregated_state, strict=True)

    def train(self):
        if self.is_delayed and self._teacher_model is not None:
            self._kd_train()
        else:
            self._local_train()
        self.neighbor_model_weights.clear()

    # required by Coordinator ----------------------------------------------------
    def send_model(self):
        return self.model.state_dict()

    def set_init_model(self, model):
        self.model = deepcopy(model)
        if (not self.is_delayed) and self.neighbor_model_weights:
            self.aggregate()

    # ────────────────────────────── KD training ────────────────────────────────
    def _kd_train(self):
        """Local training with task CE + multi‑level KD."""
        self.model.train()
        self._teacher_model.eval()

        for _ in range(self.epochs):
            for x, y in self.client_train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # -------------------- forward --------------------
                with torch.no_grad():
                    teacher_feats = self._forward_with_intermediate(self._teacher_model, x, detach=True)
                student_feats = self._forward_with_intermediate(self.model, x, detach=False)

                # ------------------- CE loss --------------------
                ce_loss = self.criterion(student_feats[-1], y).mean()

                # -------------- update category τ̃_c ------------- (very simple)
                with torch.no_grad():
                    batch_ce = F.cross_entropy(teacher_feats[-1], y, reduction="none")
                    class_ce = torch.zeros_like(self._cat_tau)
                    for c in range(self.num_classes):
                        mask = y == c
                        if mask.any():
                            class_ce[c] = batch_ce[mask].mean()
                    global_mean = class_ce[class_ce > 0].mean()
                    diff = class_ce - global_mean
                    beta = 0.05
                    self._cat_tau = torch.clamp(self._cat_tau - beta * diff, 0.5, 5.0)

                # ---------------- KD loss (multi‑level) ----------------
                L = min(len(student_feats), len(teacher_feats))
                kd_loss = 0.0
                for l in range(L):
                    # elastic τₗ : base * (1 + γ · (l / L))  ·  category τ̃  (batch avg)
                    cat_tau = self._cat_tau[y].mean().item()
                    tau_l = (self._base_tau + self._elastic_gamma * l / (L - 1 + 1e-6)) * cat_tau

                    s_feat = student_feats[l]
                    t_feat = teacher_feats[l]

                    # flatten to [B, C] for KL (if spatial)
                    if s_feat.dim() > 2:
                        s_feat = torch.flatten(s_feat, start_dim=1)
                        t_feat = torch.flatten(t_feat, start_dim=1)
                    kd_loss = kd_loss + _kl_divergence(s_feat, t_feat, tau_l)

                loss = ce_loss + kd_loss

                # ---------------- back‑prop ----------------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @staticmethod
    def _forward_with_intermediate(model: nn.Module, x: torch.Tensor, *, detach: bool = False) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        hooks = []

        def _make_hook(store):
            def fn(_mod, _inp, out):
                store.append(out.detach() if detach else out)
            return fn

        for n, m in model.named_modules():
            if n and hasattr(m, "weight") and m.weight is not None and "bn" not in n and "downsample" not in n:
                hooks.append(m.register_forward_hook(_make_hook(feats)))

        logits = model(x)

        for h in hooks:
            h.remove()

        feats.append(logits.detach() if detach else logits)
        return feats
