from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from clients.client import Client

def _log_softmax_tau(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return F.log_softmax(logits / tau, dim=-1)

def _softmax_tau(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return F.softmax(logits / tau, dim=-1)

def _kl_divergence(student: torch.Tensor,
                   teacher: torch.Tensor,
                   tau: float) -> torch.Tensor:
    """KL(teacher ‖ student) with temperature scaling."""
    log_ps = _log_softmax_tau(student, tau)
    with torch.no_grad():
        pt = _softmax_tau(teacher, tau)
    return F.kl_div(log_ps, pt, reduction="batchmean") * (tau ** 2)

def _z_score(feat: torch.Tensor) -> torch.Tensor:
    mu = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True) + 1e-6
    return (feat - mu) / std


# -------------------- Main Class -------------------- #
class ReTFHDClient(Client):

    def __init__(self,
                 client_id: int,
                 dataset_index,
                 full_dataset,
                 hyperparam,
                 device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        self.neighbor_logits: List[torch.Tensor] = []
        self.class_logits: Optional[torch.Tensor] = None
        self.global_teacher_logits: Optional[torch.Tensor] = None
        self.base_tau: float = hyperparam.get("tau", 2.0)
        self.elastic_gamma: float = hyperparam.get("gamma", 1.0)
        self.cat_tau = torch.ones(self.num_classes, device=device)

    def send_model(self):
        # returns (C, C) tensor of per-class averaged logits
        if self.class_logits is None:
            return torch.zeros(self.num_classes, self.num_classes)
        return self.class_logits.clone().cpu()

    def set_init_model(self, model):
        self.model = deepcopy(model)
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if len(self.neighbor_model_weights) != 0:
            self.aggregate()

    def receive_neighbor_model(self, logits_tensor):
        # collect neighbors' logits
        self.neighbor_logits.append(logits_tensor.to(self.device))

    def aggregate(self):
        if not self.neighbor_logits:
            return
        self.global_teacher_logits = torch.stack(self.neighbor_logits).mean(dim=0)
        self.neighbor_logits.clear()

    def train(self):
        if self.global_teacher_logits is not None:
            self._kd_train()
        else:
            self._local_train()
        self.class_logits = self._compute_class_logits()

    def _compute_class_logits(self) -> torch.Tensor:
        self.model.eval()
        agg = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        cnt = torch.zeros(self.num_classes, device=self.device)
        with torch.no_grad():
            for x, y in self.client_train_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)  # (B, C)
                for c in range(self.num_classes):
                    mask = (y == c)
                    if mask.any():
                        agg[c] += out[mask].mean(dim=0)
                        cnt[c] += 1
        cnt = torch.where(cnt == 0, torch.ones_like(cnt), cnt)
        return agg / cnt.unsqueeze(1)

    def _kd_train(self):
        if not hasattr(self, "proj_heads"):
            self.proj_heads = nn.ModuleDict().to(self.device)

        def _z_score(t: torch.Tensor) -> torch.Tensor:
            mu = t.mean(dim=1, keepdim=True)
            var = t.var(dim=1, unbiased=False, keepdim=True) + 1e-6
            return (t - mu) / var.sqrt()

        self.model.train()
        for _ in range(self.epochs):
            for x, y in self.client_train_loader:
                x, y = x.to(self.device), y.to(self.device)

                student_feats = self._forward_with_intermediate(self.model, x, detach=False)

                teacher_logits = self.global_teacher_logits[y]

                ce_loss = self.criterion(student_feats[-1], y).mean()

                kd_loss = torch.zeros(1, device=self.device)
                with torch.no_grad():
                    z_t = _z_score(teacher_logits)

                for idx, feat in enumerate(student_feats):
                    feat_flat = feat
                    if feat_flat.dim() > 2:
                        feat_flat = torch.flatten(feat_flat, 1)
                    lname = f"layer{idx}"
                    if lname not in self.proj_heads:
                        self.proj_heads[lname] = nn.Linear(feat_flat.size(1), self.num_classes).to(self.device)
                    s_logits = self.proj_heads[lname](feat_flat)

                    # Z‑Score
                    z_s = _z_score(s_logits)
                    delta_z = (z_t - z_s).abs().mean()
                    tau_l = max(self.base_tau * (1 + self.elastic_gamma * delta_z.item()), 1e-3)

                    kd_loss = kd_loss + _kl_divergence(s_logits, teacher_logits, tau_l)

                loss = ce_loss + kd_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @staticmethod
    def _forward_with_intermediate(model: nn.Module, x: torch.Tensor, *, detach: bool = False) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        hooks: List = []
        def _make_hook(store):
            def fn(_mod, _inp, out):
                store.append(out.detach() if detach else out)
            return fn
        for n, m in model.named_modules():
            if n and hasattr(m, 'weight') and m.weight is not None \
               and 'bn' not in n and 'downsample' not in n and 'conv' not in n:
                hooks.append(m.register_forward_hook(_make_hook(feats)))
        _ = model(x)
        for h in hooks:
            h.remove()
        out = model(x)
        feats.append(out.detach() if detach else out)
        return feats