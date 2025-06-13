from copy import deepcopy

import torch
from clients.client import Client

class DFedSAMClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        # Sharpness-Aware Minimization radius
        self.rho = hyperparam.get('rho', 0.05)

    def set_init_model(self, model):
        self.model = deepcopy(model)
        if len(self.neighbor_model_weights) > 0:
            self.aggregate()

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)

                # First forward-backward pass
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()

                # Collect gradients and build perturbation
                grad_norm = torch.norm(
                    torch.stack([p.grad.norm(p=2) for p in self.model.parameters()]), p=2
                )
                scale = self.rho / (grad_norm + 1e-12)
                # Save original parameters and apply perturbation
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    delta = p.grad * scale
                    p.data.add_(delta)

                # Second forward-backward pass at perturbed weights
                self.optimizer.zero_grad()
                outputs_perturbed = self.model(x)
                loss_perturbed = self.criterion(outputs_perturbed, labels).mean()
                loss_perturbed.backward()

                # Restore original weights by undoing perturbation
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    delta = p.grad * scale
                    p.data.sub_(delta)

                # Final optimizer step with SAM gradient
                self.optimizer.step()

        # Clear collected neighbor weights for next round
        self.neighbor_model_weights.clear()

    def aggregate(self):
        avg_weights = self._weight_aggregation()
        self.model.load_state_dict(avg_weights)
        self.neighbor_model_weights.clear()

    def send_model(self):
        return self.model.state_dict()
