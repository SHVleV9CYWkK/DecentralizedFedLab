from copy import deepcopy

import torch
import torch.nn.functional as F
from clinets.client import Client
from models.dkm import MultiTeacherDKMLayer
from utils.kmeans import TorchKMeans


class DFedCADClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        self.lambda_alignment = hyperparam['lambda_alignment'] if 'lambda_alignment' in hyperparam else 0.1
        self.n_clusters = hyperparam['n_clusters'] if 'n_clusters' in hyperparam else 16
        self.base_decay_rate = hyperparam['base_decay_rate'] if 'base_decay_rate' in hyperparam else 0.5

        self.teacher_info_list = []
        self.dkm_layers = {}
        self.mask = {}

        self.cluster_model = None
        self.global_model = None
        self.is_align = False
        self.teacher_model = None

    def _register_dkm_layers(self):
        self.dkm_layers = {}
        for key in self.model.state_dict().keys():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key and 'conv' in key:
                self.dkm_layers[key] = MultiTeacherDKMLayer(n_clusters=self.n_clusters,
                                                            alpha_mix=0.7, beta_dist=2.0).to(self.device)

    def _collect_logits_for_distill(self, model_state_dict):
        all_logits = []
        total_loss = 0.0
        total_samples = 0
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        if self.teacher_model is None:
            self.teacher_model = deepcopy(self.model)
        self.teacher_model.load_state_dict(model_state_dict)

        self.teacher_model.eval()
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(self.client_train_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                logits = self.teacher_model(x)
                loss_batch = criterion(logits, labels)
                total_loss += loss_batch.item()
                total_samples += x.size(0)
                all_logits.append(logits)

        avg_loss = total_loss / (total_samples + 1e-12)
        return torch.cat(all_logits, dim=0), avg_loss

    def _all_teacher_info(self):
        if self.lambda_alignment == 0.0:
            return
        teacher_info_list = []
        # 邻居返回三元组 教师压缩模型， 教师质心， 教师索引表
        for teacher_m, teacher_c, teacher_cl in self.neighbor_model_weights:
            teacher_logits, teacher_loss = self._collect_logits_for_distill(teacher_m)
            teacher_info_list.append({
                'centroids': teacher_c,
                'teacher_logits': teacher_logits,
                'loss': teacher_loss,
                'teacher_centroids_label': teacher_cl
            })
        beta = 0.5
        losses = torch.tensor([t["loss"] for t in teacher_info_list])
        alphas = torch.softmax(-beta * losses, dim=0)
        for t, info in enumerate(teacher_info_list):
            info["alpha"] = alphas[t].item()

        self.teacher_info_list = teacher_info_list

    def _prune_model_weights(self):
        pruned_state_dict = {}
        for key, weight in self.model.state_dict().items():
            if key in self.mask:
                pruned_state_dict[key] = weight * self.mask[key]
            else:
                pruned_state_dict[key] = weight
        return pruned_state_dict


    def _cluster_and_prune_model_weights(self):
        clustered_state_dict = {}
        mask_dict = {}
        centroids_dict = {}
        labels_dict = {}
        for key, weight in self.model.state_dict().items():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key:
                original_shape = weight.shape
                kmeans = TorchKMeans(n_clusters=self.n_clusters, is_sparse=True)
                flattened_weights = weight.detach().view(-1, 1)
                kmeans.fit(flattened_weights)
                new_weights = kmeans.centroids[kmeans.labels_].view(original_shape)
                clustered_state_dict[key] = new_weights.clone()
                centroids_dict[key] = kmeans.centroids.clone()
                labels_dict[key] = kmeans.labels_.clone()
            else:
                clustered_state_dict[key] = weight
                mask_dict[key] = torch.ones_like(weight, dtype=torch.bool)
        self.mask = mask_dict
        return clustered_state_dict, centroids_dict, labels_dict

    def _weight_aggregation(self):
        average_weights = {}
        neighbor_model_weights = [model_weights for model_weights, _ , _ in self.neighbor_model_weights]
        for key in neighbor_model_weights[0].keys():
            weighted_sum = sum(neighbor_model_weights[i][key].to(self.device) for i in range(len(neighbor_model_weights)))
            average_weights[key] = weighted_sum / len(neighbor_model_weights)

        return average_weights

    def _compute_global_local_model_difference(self):
        global_dict = self.global_model.state_dict()
        local_dict = self.model.state_dict()
        difference_dict = {}
        for key in global_dict:
            difference_dict[key] = local_dict[key] - global_dict[key]
        return difference_dict

    def _compute_alignment_loss(self):
        losses = []
        # 多教师质心 & labels 都在 self.teacher_info_list
        for layer_key, dkm in self.dkm_layers.items():
            # 1) 拿学生当前权重
            W = self.model.state_dict()[layer_key].to(self.device)  # Tensor
            Wf = W.view(-1, 1)  # (N_w,1)

            # 2) 准备教师输入
            teacher_centroids = torch.stack(
                [t["centroids"][layer_key].to(self.device)
                 for t in self.teacher_info_list], dim=0  # (T,K,1)
            )
            teacher_alphas = torch.tensor([t["alpha"] for t in self.teacher_info_list], device=self.device)  # (T,)

            # 3) 调用 DKM 层，得到重构
            X_rec, _, _ = dkm(
                Wf,
                teacher_centroids=teacher_centroids,
                teacher_alphas=teacher_alphas,
                teacher_index_tables=None,
                lambda_teacher=self.lambda_alignment
            )

            # 4) 重构误差
            losses.append(F.mse_loss(Wf, X_rec))

        if losses:
            return torch.stack(losses).sum()
        else:
            return torch.zeros((), device=self.device)

    def aggregate(self):
        self.global_model.load_state_dict(self._weight_aggregation())
        self.neighbor_model_weights.clear()


    def set_init_model(self, model):
        self.model = deepcopy(model)
        self.global_model = deepcopy(model)
        self.is_align = len(self.neighbor_model_weights) != 0 and self.lambda_alignment != 0

    def train(self):
        if self.is_align and len(self.dkm_layers) == 0:
            self._register_dkm_layers()

        if self.neighbor_model_weights and self.is_align:
            self._all_teacher_info()
            self.aggregate()

        for epoch in range(self.epochs):
            ref_momentum = self._compute_global_local_model_difference()

            self.model.train()

            exponential_average_loss = None
            alpha = 0.5

            for batch_idx, (x, labels) in enumerate(self.client_train_loader):
                self.model.load_state_dict(self._prune_model_weights())

                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss_sup = self.criterion(outputs, labels).mean()

                loss_align = self._compute_alignment_loss()

                loss_final = loss_sup + self.lambda_alignment * loss_align
                loss_final.backward()

                if exponential_average_loss is None:
                    exponential_average_loss = loss_final.item()
                else:
                    exponential_average_loss = alpha * loss_final.item() + (1 - alpha) * exponential_average_loss

                if loss_final.item() < exponential_average_loss:
                    decay_factor = min(self.base_decay_rate ** (batch_idx + 1) * 1.1, 0.8)
                else:
                    decay_factor = max(self.base_decay_rate ** (batch_idx + 1) / 1.1, 0.1)

                for name, param in self.model.named_parameters():
                    if name in ref_momentum:
                        param.grad += decay_factor * ref_momentum[name]
                self.optimizer.step()

        self.cluster_model = self._cluster_and_prune_model_weights()

    def send_model(self):
        return self.cluster_model
