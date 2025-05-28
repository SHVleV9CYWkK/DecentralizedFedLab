from copy import deepcopy

from clinets.client import Client

class DFedAvgClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

    def aggregate(self):
        self.model.load_state_dict(self._weight_aggregation())
        self.neighbor_model_weights.clear()

    def set_init_model(self, model):
        self.model = deepcopy(model)
        if len(self.neighbor_model_weights) != 0:
            self.aggregate()

    def train(self):
        self._local_train()

    def send_model(self):
        return self.model.state_dict()
