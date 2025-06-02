from copy import deepcopy

import torch

from clinets.client import Client


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
        all_keys = list(model.state_dict().keys())
        self.shared_keys = [k for k in all_keys if "shared" in k]
        self.personal_keys = [k for k in all_keys if "personal" in k or "classifier" in k]

        # Initial biased model (u): full state dict with only shared params
        self.u = {k: v.clone().detach().to(self.device) for k, v in self.model.state_dict().items() if k in self.shared_keys}
        self.z = {k: v.clone().detach().to(self.device) for k, v in self.u.items()}
        self.v = {k: v.clone().detach().to(self.device) for k, v in self.model.state_dict().items() if k in self.personal_keys}

    def train(self):
        self.model.train()

        for _ in range(self.Kv):  # update personal part
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self._load_weights_to_model(shared=self.z, personal=self.v)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                self.optimizer.step()
                self.v = {k: v.clone().detach() for k, v in self.model.state_dict().items() if k in self.personal_keys}

        for _ in range(self.Ku):  # update shared part (biased)
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self._load_weights_to_model(shared=self.z, personal=self.v)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                self.optimizer.step()
                self.u = {k: v.clone().detach() for k, v in self.model.state_dict().items() if k in self.shared_keys}
                self.z = {k: self.u[k] / self.mu for k in self.u}

    def _load_weights_to_model(self, shared, personal):
        state_dict = self.model.state_dict()
        for k in self.shared_keys:
            state_dict[k] = shared[k].clone()
        for k in self.personal_keys:
            state_dict[k] = personal[k].clone()
        self.model.load_state_dict(state_dict, strict=False)

    def send_model(self):
        return {k: self.u[k] * 1.0 for k in self.shared_keys}, self.mu

    def aggregate(self):
        if not self.neighbor_model_weights:
            return

        # Push-sum aggregation
        aggregated_u = {k: torch.zeros_like(v) for k, v in self.u.items()}
        aggregated_mu = 0.0

        for u_j, mu_j in self.neighbor_model_weights:
            for k in aggregated_u:
                aggregated_u[k] += u_j[k]
            aggregated_mu += mu_j

        self.u = aggregated_u
        self.mu = aggregated_mu
        self.z = {k: self.u[k] / self.mu for k in self.u}
        self.neighbor_model_weights.clear()
