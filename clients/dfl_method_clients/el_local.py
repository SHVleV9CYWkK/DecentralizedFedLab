from copy import deepcopy

from clients.client import Client


class ELLocalClient(Client):
    """
    Epidemic Learning - Local sampling variant

    论文含义：
    - 每轮先做本地训练，得到 x_{t+1/2}
    - 每个客户端独立随机选 s 个其他客户端发送自己的模型
    - 每个客户端用 “自己的当前模型 + 收到的模型” 做等权平均
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

    def _el_local_aggregation(self):
        """
        对应论文公式 (2):
        x_i^{t+1} = (x_i^{t+1/2} + sum(received_models)) / (|S_i^t| + 1)

        注意：
        - 必须把“自己当前模型”也纳入平均
        - 即使一个邻居模型都没收到，也应保留自己的模型
        """
        if self.model is None:
            raise ValueError(f"Client {self.id}: model has not been initialized.")

        local_state = self.model.state_dict()

        aggregated_state = {
            k: v.detach().clone().to(self.device)
            for k, v in local_state.items()
        }

        for neighbor_state in self.neighbor_model_weights:
            for k in aggregated_state.keys():
                aggregated_state[k] += neighbor_state[k].to(self.device)

        divisor = len(self.neighbor_model_weights) + 1
        for k in aggregated_state.keys():
            aggregated_state[k] /= divisor

        return aggregated_state

    def aggregate(self):
        """
        EL-Local 聚合：
        平均 当前本地模型 + 所有接收到的模型
        """
        aggregated_weights = self._el_local_aggregation()
        self.model.load_state_dict(aggregated_weights)
        self.neighbor_model_weights.clear()

    def set_init_model(self, model):
        self.model = deepcopy(model)

    def train(self):
        """
        这里沿用你现有框架的本地训练逻辑。
        虽然论文 Algorithm 1 写的是每轮一次随机梯度更新，
        但你的工程是本地多步/多 epoch 训练，这里保持和现有系统一致。
        """
        self._local_train()

    def send_model(self):
        """
        返回当前模型副本，避免通信阶段共享底层 tensor 引发副作用。
        建议统一转到 CPU，减少跨设备引用问题。
        """
        return {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }