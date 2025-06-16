import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from clients.client import Client

# ===== Generator 模块 =====
class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_shape=(1, 28, 28)):
        super().__init__()
        self.noise_dim = noise_dim
        self.output_shape = output_shape
        flat_dim = int(torch.prod(torch.tensor(output_shape)))
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, flat_dim),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, *self.output_shape)

# ===== Discriminator 模块 =====
class Discriminator(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        flat_dim = int(torch.prod(torch.tensor(input_shape)))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# ===== FedGO 客户端 =====
class FedGOClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        # FedGO 参数
        self.noise_dim = hyperparam.get('noise_dim', 100)
        self.gen_epochs = hyperparam.get('gen_epochs', 3)
        self.disc_epochs = hyperparam.get('disc_epochs', 1)
        self.unlabeled_samples = hyperparam.get('unlabeled_samples', 128)
        self.lambda_kd = hyperparam.get('lambda_kd', 1.0)
        self.lambda_ce = hyperparam.get('lambda_ce', 1.0)
        # 模型与判别器、生成器
        self.generator = Generator(self.noise_dim, full_dataset[0][0].shape).to(device)
        self.discriminator = Discriminator(full_dataset[0][0].shape).to(device)
        # 邻居参数存储
        self.peer_models = {}         # id -> state_dict
        self.peer_discriminators = {} # id -> state_dict
        self.is_later = False


    def set_init_model(self, model):
        # 初始化全局模型
        self.model = deepcopy(model)
        # 预训练：generator + discriminator
        self.train_generator()
        self.train_discriminator()

        if len(self.peer_models) != 0:
            self.is_later = True

    def train(self):
        if self.is_later:
            self._local_train_kd()
        else:
            self._local_train()



    def send_model(self):
        # 发送 model 和 discriminator 状态
        return (self.id, {
            'model': deepcopy(self.model.state_dict()),
            'discriminator': deepcopy(self.discriminator.state_dict())
        })

    def receive_neighbor_model(self, neighbor_model):
        client_id, data = neighbor_model
        self.neighbor_model_weights.append(data['model'])
        self.peer_models[client_id] = data['model']
        self.peer_discriminators[client_id] = data['discriminator']

    def _local_train_kd(self):
        # FedGO 聚合：蒸馏
        # 1. 生成伪未标注数据
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(self.unlabeled_samples, self.noise_dim, device=self.device)
            fake_x = self.generator(z)
        # 2. 计算伪标签
        pseudo_labels = []  # tensor list
        for x in fake_x:
            x = x.unsqueeze(0)
            logits_sum = 0
            weight_sum = 0
            for cid, mstate in self.peer_models.items():
                # peer model forward
                peer = deepcopy(self.model).to(self.device)
                peer.load_state_dict(mstate)
                peer.eval()
                out = peer(x)
                # peer discriminator
                disc = Discriminator(x.shape[1:]).to(self.device)
                disc.load_state_dict(self.peer_discriminators[cid])
                d_out = torch.sigmoid(disc(x))
                odds = d_out / (1 - d_out + 1e-6)
                logits_sum += odds * out
                weight_sum += odds
            logits_avg = logits_sum / (weight_sum + 1e-6)
            pseudo_labels.append(logits_avg.squeeze(0))
        pseudo_labels = torch.stack(pseudo_labels)
        # 3. 蒸馏训练
        # 构建 unlabeled DataLoader
        dataset = TensorDataset(fake_x, pseudo_labels)
        loader_u = DataLoader(dataset, batch_size=self.client_val_loader.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            # KD on unlabeled
            for x_u, y_u in loader_u:
                x_u, y_u = x_u.to(self.device), y_u.to(self.device)
                self.optimizer.zero_grad()
                out_u = self.model(x_u)
                loss_kd = F.kl_div(F.log_softmax(out_u, dim=1), F.softmax(y_u.detach(), dim=1), reduction='batchmean')
                loss_kd *= self.lambda_kd
                loss_kd.backward()
                self.optimizer.step()
            # CE on real data
            for x_r, labels in self.client_train_loader:
                x_r, labels = x_r.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                out_r = self.model(x_r)
                loss_ce = F.cross_entropy(out_r, labels) * self.lambda_ce
                loss_ce.backward()
                self.optimizer.step()
        # 清空 peer buffers
        self.peer_models.clear()
        self.peer_discriminators.clear()

    def train_generator(self):
        # 本地用真实数据训练生成器
        self.generator.train()
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        for _ in range(self.gen_epochs):
            for x, _ in self.client_train_loader:
                x = x.to(self.device)
                z = torch.randn(x.size(0), self.noise_dim, device=self.device)
                fake = self.generator(z)
                loss = F.mse_loss(fake, x)
                opt_g.zero_grad()
                loss.backward()
                opt_g.step()

    def aggregate(self):
        self.model.load_state_dict(self._weight_aggregation())
        self.neighbor_model_weights.clear()

    def train_discriminator(self):
        # 本地训练判别器分辨真实 vs 生成
        self.discriminator.train()
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        bce = nn.BCEWithLogitsLoss()
        for _ in range(self.disc_epochs):
            for x, _ in self.client_train_loader:
                x = x.to(self.device)
                # real
                real_label = torch.ones(x.size(0), 1, device=self.device)
                out_real = self.discriminator(x)
                loss_real = bce(out_real, real_label)
                # fake
                z = torch.randn(x.size(0), self.noise_dim, device=self.device)
                fake_x = self.generator(z).detach()
                fake_label = torch.zeros(x.size(0), 1, device=self.device)
                out_fake = self.discriminator(fake_x)
                loss_fake = bce(out_fake, fake_label)
                # update
                loss = (loss_real + loss_fake) * 0.5
                opt_d.zero_grad()
                loss.backward()
                opt_d.step()
