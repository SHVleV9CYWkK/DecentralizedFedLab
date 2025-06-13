from copy import deepcopy

import torch

from clients.client import Client


class DFedPGPClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        self.shared_keys = None
        self.personal_keys = None
        self.mu = 1.0
        self.u = None
        self.z = None
        self.v = None
        self.out_weights = []
        self.Ku = hyperparam.get("Ku", 1)
        self.Kv = hyperparam.get("Kv", 1)

    def set_init_model(self, model):
        self.model = deepcopy(model)
        self.model.to(self.device)

        state_dict = self.model.state_dict()
        module_names = list(self.model.named_modules())

        # 找到第一个出现 Linear 的模块名位置
        linear_start_index = None
        for i, (name, module) in enumerate(module_names):
            if isinstance(module, torch.nn.Linear):
                linear_start_index = i
                break

        if linear_start_index is None:
            raise ValueError("Cannot find any Linear layers for personalization.")

        # 从第一个 Linear 开始，所有后续模块名都当作个性化部分
        personal_prefixes = [name for name, _ in module_names[linear_start_index:] if name != '']

        # 标记哪些 state_dict 的 key 属于个人化
        self.personal_keys = [
            k for k in state_dict.keys()
            if any(k.startswith(prefix) for prefix in personal_prefixes)
        ]
        self.shared_keys = [k for k in state_dict.keys() if k not in self.personal_keys]

        self.u = {k: state_dict[k].clone().detach().to(self.device) for k in self.shared_keys}
        self.z = {k: v.clone() for k, v in self.u.items()}
        self.v = {k: state_dict[k].clone().detach().to(self.device) for k in self.personal_keys}

    def train(self):
        self.model.train()
        self._load_weights_to_model(shared=self.z)

        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                self.optimizer.step()
                self.v = {k: v.clone().detach() for k, v in self.model.state_dict().items() if k in self.personal_keys}

    def _load_weights_to_model(self, shared):
        state_dict = self.model.state_dict()
        for k in self.shared_keys:
            state_dict[k] = shared[k].clone()
        self.model.load_state_dict(state_dict)

    def send_model(self):
        return {k: self.u[k] * 1.0 for k in self.shared_keys}, self.mu

    def aggregate(self):
        if not self.neighbor_model_weights:
            return

        # Push-sum aggregation
        aggregated_u = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.u.items()}
        aggregated_mu = 0.0

        for u_j, mu_j in self.neighbor_model_weights:
            for k in aggregated_u:
                aggregated_u[k] += u_j[k]
            aggregated_mu += mu_j

        self.u = aggregated_u
        self.mu = aggregated_mu
        self.z = {k: self.u[k] / self.mu for k in self.u}
        self.neighbor_model_weights.clear()
