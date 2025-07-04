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


def _kl_divergence(student_logits: torch.Tensor,
                   teacher_logits: torch.Tensor,
                   tau: float) -> torch.Tensor:
    log_ps = _log_softmax_tau(student_logits, tau)  # log‑prob student
    with torch.no_grad():
        pt = _softmax_tau(teacher_logits, tau)      # prob teacher
    return F.kl_div(log_ps, pt, reduction="batchmean") * (tau ** 2)

class ReTFHDClient(Client):

    def __init__(self,
                 client_id: int,
                 dataset_index,
                 full_dataset,
                 hyperparam,
                 device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

        self.is_delayed: bool = False
        self._teacher_model: Optional[nn.Module] = None

        self._cat_tau = torch.ones(self.num_classes, device=device)

        self._base_tau: float = hyperparam.get("tau", 2.0)
        self._elastic_gamma: float = hyperparam.get("gamma", 1.0)

        self._hook_handles = []

    def init_client(self):
        super().init_client()
        if len(self.neighbor_model_weights) != 0:
            self.is_delayed = True
            self.aggregate()

    def aggregate(self):
        if not self.neighbor_model_weights:
            return
        aggregated_state = self._weight_aggregation()

        if self.is_delayed:
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

    def send_model(self):
        return self.model.state_dict()

    def set_init_model(self, model):
        self.model = deepcopy(model)
        if (not self.is_delayed) and self.neighbor_model_weights:
            self.aggregate()

    def _kd_train(self):
        self.model.train()
        self._teacher_model.eval()

        for _ in range(self.epochs):
            for x, y in self.client_train_loader:
                x, y = x.to(self.device), y.to(self.device)

                with torch.no_grad():
                    teacher_feats = self._forward_with_intermediate(self._teacher_model, x, detach=True)
                student_feats = self._forward_with_intermediate(self.model, x, detach=False)

                ce_loss = self.criterion(student_feats[-1], y).mean()

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

                L = min(len(student_feats), len(teacher_feats))
                kd_loss = 0.0
                for l in range(L):
                    cat_tau = self._cat_tau[y].mean().item()
                    tau_l = (self._base_tau + self._elastic_gamma * l / (L - 1 + 1e-6)) * cat_tau

                    s_feat = student_feats[l]
                    t_feat = teacher_feats[l]

                    if s_feat.dim() > 2:
                        s_feat = torch.flatten(s_feat, start_dim=1)
                        t_feat = torch.flatten(t_feat, start_dim=1)
                    kd_loss = kd_loss + _kl_divergence(s_feat, t_feat, tau_l)

                loss = ce_loss + kd_loss

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
