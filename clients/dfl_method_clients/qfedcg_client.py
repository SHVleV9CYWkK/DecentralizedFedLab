import torch
from torch.quantization import QuantStub, DeQuantStub, QConfig, default_observer
from copy import deepcopy

from clients.client import Client


class QFedCGClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        self.quantization_levels = kwargs.get('initial_quantization_levels', 8)
        self.l_max = kwargs.get('max_quantization_levels', 8)
        self.last_gradient = None
        self.qconfig = None
        self.quantizer = None
        self.dequantizer = None
        self.neighbor_gradients = []
        self.compression_ratio = hyperparam.get('compression_ratio', 0.5)

    def set_init_model(self, model):
        self.model = deepcopy(model).to(self.device)
        self.neighbor_gradients.clear()
        self.last_gradient = None
        self.initialize_quantization()

    def initialize_quantization(self):
        # 根据当前量化级别建立observer
        observer = default_observer.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-2 ** (self.quantization_levels - 1),
            quant_max=2 ** (self.quantization_levels - 1) - 1
        )
        self.qconfig = QConfig(activation=observer, weight=observer)
        self.quantizer = QuantStub()
        self.dequantizer = DeQuantStub()
        self.quantizer.qconfig = self.qconfig
        self.dequantizer.qconfig = self.qconfig

        torch.quantization.prepare(self.quantizer, inplace=True)
        torch.quantization.convert(self.quantizer, inplace=True)
        torch.quantization.prepare(self.dequantizer, inplace=True)
        torch.quantization.convert(self.dequantizer, inplace=True)

    def get_top_k_gradients(self):
        sparse = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            flat = param.grad.abs().flatten()
            total_params = param.numel()
            k = int(total_params * self.compression_ratio)
            top_vals, top_idx = torch.topk(flat, k, largest=True, sorted=False)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[top_idx] = True
            mask = mask.view_as(param.grad)
            sparse[name] = param.grad * mask
        return sparse

    def quantize(self, tensor):
        q = self.quantizer(tensor)
        dq = self.dequantizer(q)
        return dq

    def compress_and_quantize_gradients(self):
        sparse = self.get_top_k_gradients()
        quantized = {n: self.quantize(g) for n, g in sparse.items()}
        return quantized

    def calculate_quantization_levels(self, current_grad):
        if self.last_gradient is None:
            self.last_gradient = current_grad
            return
        innovation = {}
        for name, grad in current_grad.items():
            delta = grad - self.last_gradient.get(name, torch.zeros_like(grad))
            innovation[name] = delta.norm(2)

        avg_innov = sum(innovation.values()).item() / len(innovation)
        if avg_innov > 1.0:
            self.quantization_levels = max(1, self.quantization_levels - 1)
        else:
            self.quantization_levels = min(self.l_max, self.quantization_levels + 1)
        self.last_gradient = current_grad

    def train(self):
        self.initialize_quantization()
        self._local_train()

    def send_model(self):
        quantized_grads = self.compress_and_quantize_gradients()
        self.calculate_quantization_levels({k: g.clone() for k, g in quantized_grads.items()})
        return quantized_grads

    def receive_neighbor_model(self, neighbor_grad):
        self.neighbor_gradients.append(neighbor_grad)

    def aggregate(self):
        if not self.neighbor_gradients:
            return

        agg = {}
        n = len(self.neighbor_gradients)
        for grad_dict in self.neighbor_gradients:
            for k, v in grad_dict.items():
                if k not in agg:
                    agg[k] = v.clone().to(self.device)
                else:
                    agg[k] += v.to(self.device)
        for k in agg:
            agg[k] /= n
        for name, param in self.model.named_parameters():
            if name in agg:
                param.grad = agg[name]
            else:
                param.grad = torch.zeros_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.neighbor_gradients.clear()

