from copy import deepcopy

from clients.client import Client


class LocalOnlyClient(Client):
    """Local-only 基线：客户端只训练自己的数据，完全不参与任何模型交换。

    作用（论文动机的下界检验）：给出"完全不协作"的参考线——协作（DFL）相对
    单练到底值多少？对延迟加入场景尤其直接：新客户端加入网络 vs 自己单练，
    哪个更好？在强 non-IID（本地验证集与本地标签分布对齐）下这条线常常不低，
    因此是必要的对照。

    实现：aggregate() 丢弃收到的一切邻居模型（Coordinator 仍按拓扑分发，
    本类不使用），故每个客户端的轨迹是独立的本地 SGD。延迟客户端在 τ_k 处
    以 common_init 开始本地训练（无热启动可言——它不接收任何模型）。
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

    def set_init_model(self, model):
        # 不做任何加入时聚合（对比 DFedAvgClient：那里会 aggregate 收到的邻居模型）
        self.model = deepcopy(model)
        self.neighbor_model_weights.clear()

    def train(self):
        self._local_train()

    def send_model(self):
        # 仍需返回模型：Coordinator 会按拓扑分发给邻居；本方法的客户端一律丢弃
        return self.model.state_dict()

    def aggregate(self):
        # 核心：不混合。丢弃全部收到的邻居模型，保持纯本地轨迹
        self.neighbor_model_weights.clear()
