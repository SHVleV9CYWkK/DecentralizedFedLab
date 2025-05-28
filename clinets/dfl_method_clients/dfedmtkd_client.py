from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from clinets.client import Client


def find_last_linear_and_prev(model):
    last_linear = None
    prev_layer = None
    prev = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prev_layer = prev
            last_linear = module
            break
        prev = module
    if last_linear is None:
        raise ValueError("No Linear layer found in the model!")
    return prev_layer, last_linear


class DFedMTKDClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        self.lambda_logit_kd = hyperparam.get('lambda_kd', 0.1)
        self.lambda_feature_kd = hyperparam.get('lambda_feature_kd', 0.1)
        self._student_hook_handle = None
        self.features_buffer = []
        self.teacher_info_list = []
        self.teacher_model = None
        self.teacher_features_outputs = []
        self._teacher_hook_handle = None

    def _student_hook_fn(self, module, input, output):
        self.features_buffer.append(output.detach())

    def _teacher_hook_fn(self, module, input, output):
        self.teacher_features_outputs.append(output.detach())

    def train(self):
        num_teachers = len(self.neighbor_model_weights)

        for epoch in range(self.epochs):
            self.model.train()
            for x, labels in self.client_train_loader:
                x = x.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)  # 避免额外填 0
                self.features_buffer.clear()

                stu_logits = self.model(x)
                loss_sup = self.criterion(stu_logits, labels).mean()

                loss_logit_kd = loss_feature_kd = 0.0
                if num_teachers and (self.lambda_logit_kd or self.lambda_feature_kd):
                    with torch.inference_mode():  # hook 写入 teacher_features_outputs
                        self.teacher_features_outputs.clear()
                        tea_logits = self.teacher_model(x)
                        tea_feat = self.teacher_features_outputs[-1]
                        self.teacher_features_outputs.clear()

                    if self.lambda_logit_kd:
                        loss_logit_kd = F.kl_div(
                            F.log_softmax(stu_logits, 1),
                            F.softmax(tea_logits, 1),
                            reduction='batchmean'
                        )
                    if self.lambda_feature_kd:
                        loss_feature_kd = F.mse_loss(self.features_buffer[0], tea_feat)

                loss = (loss_sup
                        + self.lambda_logit_kd * loss_logit_kd
                        + self.lambda_feature_kd * loss_feature_kd)

                loss.backward()
                self.optimizer.step()

    def send_model(self):
        return self.model.state_dict()

    def aggregate(self):
        pass

    def set_init_model(self, model):
        self.model = deepcopy(model)
        prev_layer, _ = find_last_linear_and_prev(self.model)
        self._student_hook_handle = prev_layer.register_forward_hook(self._student_hook_fn)

        self.teacher_model = deepcopy(model).eval()
        prev_layer, _ = find_last_linear_and_prev(self.teacher_model)
        self._teacher_hook_handle = prev_layer.register_forward_hook(self._teacher_hook_fn)
