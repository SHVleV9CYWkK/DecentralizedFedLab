from copy import deepcopy

import torch
import torch.nn.functional as F
from clinets.client import Client
from models.dkm import MultiTeacherDKMLayer
from utils.kmeans import TorchKMeans


def _cfd_distance(centroids_a, centroids_b, n_freqs=512, sigma=1.0):
    device = centroids_a.device
    if centroids_a.ndim == 1:
        centroids_a = centroids_a.view(-1, 1)
        centroids_b = centroids_b.view(-1, 1)
    D = centroids_a.shape[1]
    # 采样 n_freqs 个频率向量
    freqs = torch.randn(n_freqs, D, device=device) * sigma  # [n_freqs, D]
    # [n_freqs, K] ← [n_freqs, D] x [K, D]^T
    fa = (freqs @ centroids_a.T)   # shape: [n_freqs, K]
    fb = (freqs @ centroids_b.T)
    phi_a = torch.mean(torch.exp(1j * fa), dim=1)  # [n_freqs]
    phi_b = torch.mean(torch.exp(1j * fb), dim=1)  # [n_freqs]
    cfd = torch.mean(torch.abs(phi_a - phi_b) ** 2)
    return cfd.item() if not isinstance(cfd, float) else cfd


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

    def _all_teacher_info(self):
        # 聚类本地模型，获得本地质心表
        _, local_centroids_dict, _ = self._cluster_and_prune_model_weights()
        cfd_scores = []
        teacher_centroids_dicts = []
        cfd_matrix = []
        # 遍历每个教师模型：三元组(权重, centroids, 索引)
        for _, teacher_centroids, _ in self.neighbor_model_weights:
            per_layer_cfd = []
            teacher_centroids_dicts.append(teacher_centroids)
            # 多层时建议做“加权平均或拼接后再算 CFD”
            for layer_key in local_centroids_dict:
                cfd = _cfd_distance(
                    local_centroids_dict[layer_key].detach().float(),
                    teacher_centroids[layer_key].detach().float()
                )
                per_layer_cfd.append(cfd)
            cfd_matrix.append(per_layer_cfd)
        cfd_matrix_tensor = torch.tensor(cfd_matrix, dtype=torch.float)
        cfd_scores = torch.mean(cfd_matrix_tensor, dim=1)

        # 归一化为权重
        cfd_tensor = torch.tensor(cfd_scores, dtype=torch.float)
        min_val = cfd_tensor.min()
        max_val = cfd_tensor.max()
        normed = (cfd_tensor - min_val) / (max_val - min_val + 1e-8)

        beta = 2 # 可调节的“温度”参数，控制softmax分布
        alphas = torch.softmax(-beta * normed, dim=0)
        # 存储
        self.teacher_info_list = [
            {
                'centroids': teacher_centroids_dicts[i],
                'alpha': alphas[i].item()
            }
            for i in range(len(teacher_centroids_dicts))
        ]

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
        if len(self.teacher_info_list) == 0:
            return torch.zeros((), device=self.device)

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

    def _local_train(self):
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

    def aggregate(self):
        self.global_model.load_state_dict(self._weight_aggregation())


    def set_init_model(self, model):
        self.model = deepcopy(model)
        self.global_model = deepcopy(model)
        self.is_align = len(self.neighbor_model_weights) != 0 and self.lambda_alignment != 0

    def train(self):
        if self.is_align and len(self.dkm_layers) == 0:
            self._register_dkm_layers()
            self._local_train()

        if self.is_align:
            self._all_teacher_info()
            self.aggregate()

        for epoch in range(self.epochs):
            self._local_train()

        self.cluster_model = self._cluster_and_prune_model_weights()
        self.neighbor_model_weights.clear()

    def send_model(self):
        return self.cluster_model
