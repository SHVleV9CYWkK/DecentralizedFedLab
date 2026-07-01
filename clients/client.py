from abc import ABC, abstractmethod

import numpy as np
import torch
import torcheval.metrics.functional as metrics
from torch.utils.data import DataLoader, Subset
from utils.utils import get_optimizer, get_lr_scheduler


class Client(ABC):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        self.id = client_id
        self.model = None
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.optimizer_name = hyperparam['optimizer_name']
        self.optimizer = None
        self.lr = hyperparam['lr']
        self.epochs = hyperparam['local_epochs']
        self.scheduler_name = hyperparam['scheduler_name']
        self.n_rounds = hyperparam['n_rounds']
        self.device = device
        train_indices = np.load(dataset_index['train']).tolist()
        val_indices = np.load(dataset_index['val']).tolist()
        self.train_dataset_len = len(train_indices)
        self.val_dataset_len = len(val_indices)
        self.num_classes = len(full_dataset.classes)
        client_train_dataset = Subset(full_dataset, indices=train_indices)
        client_val_dataset = Subset(full_dataset, indices=val_indices)
        # 默认 num_workers=0（内存型数据集 CIFAR/EMNIST 加载已很快）；工厂对 tiny_imagenet
        # （ImageFolder，磁盘 JPEG 解码）传入 >0 使加载与计算重叠。
        # 仅在 CUDA 上启用：CUDA=Linux/fork，worker 极廉价且能重叠；MPS/CPU=macOS/spawn，
        # 每个 worker 启动要重新 import torch，50 客户端反复 spawn 反而灾难性变慢。
        # 不用 persistent_workers：50 个客户端各自的 loader 会各留常驻 worker（50×N 进程爆炸）；
        # 串行训练（n_job=1）下每次只有一个客户端在跑，非常驻峰值仅 N 个 worker。
        num_workers = hyperparam.get('num_workers', 0)
        loader_kwargs = {}
        if num_workers > 0 and device.type == 'cuda':
            loader_kwargs = {'num_workers': num_workers, 'pin_memory': True}
        self.client_train_loader = DataLoader(client_train_dataset, batch_size=hyperparam['bz'], shuffle=False,
                                              drop_last=True, **loader_kwargs)
        self.client_val_loader = DataLoader(client_val_dataset,
                                            batch_size=hyperparam['bz']
                                            if hyperparam['bz'] <= len(client_val_dataset) else len(client_val_dataset),
                                            shuffle=False, **loader_kwargs)
        self.global_metric = self.global_epoch = 0
        self.lr_scheduler = None
        self.neighbor_model_weights = []
        self.last_accuracy = None

    def _weight_aggregation(self):
        average_weights = {}
        for key in self.neighbor_model_weights[0].keys():
            weighted_sum = sum(self.neighbor_model_weights[i][key].to(self.device) for i in range(len(self.neighbor_model_weights)))
            average_weights[key] = weighted_sum / len(self.neighbor_model_weights)

        return average_weights

    def _local_train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                self.optimizer.step()

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def send_model(self):
        raise NotImplementedError

    @abstractmethod
    def aggregate(self):
        raise NotImplementedError

    @abstractmethod
    def set_init_model(self, model):
        raise NotImplementedError

    def init_client(self):
        self.optimizer = get_optimizer(self.optimizer_name, self.model.parameters(), self.lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.scheduler_name, self.n_rounds)

    def update_lr(self):
        if self.last_accuracy is not None:
            self.lr_scheduler.step(self.last_accuracy)

    def receive_neighbor_model(self, neighbor_model):
        self.neighbor_model_weights.append(neighbor_model)

    def evaluate_model(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x).to(self.device)
                loss = self.criterion(outputs, labels)
                loss_meta_model = loss.mean()
                total_loss += loss_meta_model
                _, predicted = torch.max(outputs.data, 1)
                all_labels.append(labels)
                all_predictions.append(predicted)

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        avg_loss = total_loss / len(self.client_val_loader)
        accuracy = metrics.multiclass_accuracy(all_predictions, all_labels, num_classes=self.num_classes)
        precision = metrics.multiclass_precision(all_predictions, all_labels, num_classes=self.num_classes)
        recall = metrics.multiclass_recall(all_predictions, all_labels, num_classes=self.num_classes)
        f1 = metrics.multiclass_f1_score(all_predictions, all_labels, average="weighted", num_classes=self.num_classes)
        self.last_accuracy = accuracy
        return {
            'loss': avg_loss.cpu(),
            'accuracy': accuracy.cpu().item(),
            'precision': precision.cpu().item(),
            'recall': recall.cpu().item(),
            'f1': f1.cpu().item()
        }
