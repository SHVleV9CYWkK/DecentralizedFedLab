import random
import time
from copy import deepcopy
import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method, Manager
from tqdm import tqdm


def _execute_train_client(client, seed):
    try:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if client.device.type == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)
        client.train()
    except Exception as e:
        print(f"Error training client {client.id}: {str(e)}")

class Coordinator:
    def __init__(self, clients, model, device, client_delay_dict, args):
        self.all_clients = clients
        self.num_clients = len(self.all_clients)
        self.init_model = model
        self.device = device
        self.client_delay_dict = client_delay_dict
        self.participated_training_clients = [client for client in clients if client.id not in client_delay_dict]
        self.num_conn = args.num_conn
        self.gossip = args.gossip != 0
        self.symmetry = args.symmetry
        self.connected_graph = None
        self.seed = args.seed
        self.n_job = args.n_job
        self._init_clients()
        if (self.device.type == 'cuda' or self.device.type == 'cpu') and self.n_job > 1:
            try:
                set_start_method('spawn')
            except RuntimeError as e:
                print("Start method 'spawn' already set or error setting it: ", str(e))

    def _init_clients(self):
        print("Initializing initial clients...")
        pbar = tqdm(total=len(self.participated_training_clients))
        for client in self.participated_training_clients:
            client.set_init_model(deepcopy(self.init_model))
            client.init_client()
            pbar.update(1)
        pbar.clear()
        pbar.close()

    def _clone_and_detach(self, tensor_dict):
        if isinstance(tensor_dict, dict):
            return {k: self._clone_and_detach(v) for k, v in tensor_dict.items()}
        elif hasattr(tensor_dict, 'clone'):
            return tensor_dict.clone().detach().to('cpu')
        else:
            raise ValueError("Unsupported type for cloning and detaching")

    def _clients_train(self):
        if (self.device.type == 'cuda' or self.device.type == 'cpu') and self.n_job > 1:
            manager = Manager()
            return_dict = manager.dict()

            with Pool(processes=self.n_job) as pool:
                for client in self.participated_training_clients:
                    pool.apply_async(_execute_train_client, args=(client, deepcopy(self.seed)))

                with tqdm(total=len(self.participated_training_clients)) as pbar:
                    while True:
                        current_length = len(return_dict)
                        pbar.update(current_length - pbar.n)
                        if current_length >= len(self.participated_training_clients):
                            break
                        time.sleep(1)

                pool.close()
                pool.join()

            locals_weights = dict(return_dict)
        else:
            pbar = tqdm(total=len(self.participated_training_clients))
            locals_weights = dict()
            for client in self.participated_training_clients:
                client.train()
                pbar.update(1)
            pbar.clear()
            pbar.close()
        return locals_weights

    def _add_new_training_clients(self, current_round):
        participated_ids = {client.id for client in self.participated_training_clients}
        new_training_clients = []
        for client_id, delay_round in self.client_delay_dict.items():
            if delay_round == current_round:
                if client_id in participated_ids:
                    print(f"Client {client_id} has started training before adding")
                    continue
                client = self.all_clients[client_id]
                if client is None:
                    print(f"Warning: Client {client_id} not found in all_clients!")
                    continue
                # 添加客户端，并同步更新 participated_ids 集合
                self.participated_training_clients.append(client)
                client.set_init_model(deepcopy(self.init_model))
                client.init_client()
                new_training_clients.append(client_id)
                participated_ids.add(client_id)
        return new_training_clients

    def generate_connected_graph(self):
        if not self.gossip and self.connected_graph is not None:
            return

        graph = [[0 for _ in range(self.num_clients)] for _ in range(self.num_clients)]
        if self.symmetry != 0:
            print("Generating a symmetrical connectivity diagram")
            # 参数检查
            if self.num_conn > self.num_clients - 1:
                raise ValueError("For undirected graphs, each node can connect up to num_clients - 1 node")
            if (self.num_clients * self.num_conn) % 2 != 0:
                raise ValueError("For undirected graphs, num_clients * num_conn must be even")

            degree = [0] * self.num_clients

            # 1. 生成随机生成树
            nodes = list(range(self.num_clients))
            random.shuffle(nodes)
            for i in range(self.num_clients - 1):
                u = nodes[i]
                v = nodes[i+1]
                graph[u][v] = graph[v][u] = 1
                degree[u] += 1
                degree[v] += 1

            total_edges_target = (self.num_clients * self.num_conn) // 2
            current_edges = self.num_clients - 1

            # 2. 优化补充边的方法
            # 维护可用节点列表（度数未满的节点）
            available_nodes = [i for i in range(self.num_clients) if degree[i] < self.num_conn]

            while current_edges < total_edges_target and len(available_nodes) >= 2:
                # 随机选择两个不同的可用节点
                u, v = random.sample(available_nodes, 2)

                if graph[u][v] == 0:  # 如果边不存在
                    graph[u][v] = graph[v][u] = 1
                    degree[u] += 1
                    degree[v] += 1
                    current_edges += 1

                    # 更新可用节点列表
                    available_nodes = [i for i in range(self.num_clients) if degree[i] < self.num_conn]

            if current_edges < total_edges_target:
                # 如果随机方法卡住，使用确定性方法补充剩余边
                for u in range(self.num_clients):
                    for v in range(u+1, self.num_clients):
                        if degree[u] < self.num_conn and degree[v] < self.num_conn and graph[u][v] == 0:
                            graph[u][v] = graph[v][u] = 1
                            degree[u] += 1
                            degree[v] += 1
                            current_edges += 1
                            if current_edges == total_edges_target:
                                self.connected_graph = graph
            self.connected_graph = graph
        else:
            print("Generating a asymmetric connectivity diagram")
            # 有向图优化
            if self.num_conn >= self.num_clients:
                raise ValueError("For directed graphs, the out-of-out degree of each node "
                                 "must be less than num_clients (self-looping is not allowed)")

            outdegree = [0] * self.num_clients

            # 1. 生成有向生成树
            nodes = list(range(self.num_clients))
            random.shuffle(nodes)
            for i in range(self.num_clients - 1):
                u = nodes[i]
                v = nodes[i+1]
                graph[u][v] = 1
                outdegree[u] += 1

            # 2. 优化补充出边的方法
            for u in range(self.num_clients):
                # 创建可用目标节点列表
                available_targets = [v for v in range(self.num_clients)
                                     if v != u and graph[u][v] == 0]
                random.shuffle(available_targets)

                needed = self.num_conn - outdegree[u]
                for v in available_targets[:needed]:
                    graph[u][v] = 1
                    outdegree[u] += 1
            self.connected_graph = graph

    def train_client(self, current_round):
        new_clients = self._add_new_training_clients(current_round)
        if len(new_clients) != 0:
            print(f"New clients: {new_clients}")
        print("Training models...")
        self._clients_train()

    def interchange_model(self, current_round):
        for client in self.all_clients:
            client.neighbor_model_weights.clear()

        self.generate_connected_graph()
        pre_add_clients = []
        for i in range(self.num_clients):
            client_i = self.all_clients[i]

            if client_i.id in self.client_delay_dict and current_round + 1 == self.client_delay_dict[client_i.id]:
                pre_add_clients.append(i)

            if client_i.id in self.client_delay_dict and current_round < self.client_delay_dict[client_i.id]:
                continue
            for j in range(self.num_clients):
                client_j = self.all_clients[j]
                if client_j.id in self.client_delay_dict and current_round < self.client_delay_dict[client_j.id]:
                    continue
                if self.connected_graph[client_i.id][client_j.id]:
                    client_i.receive_neighbor_model(client_j.send_model())

        for i in pre_add_clients:
            client_i = self.all_clients[i]
            for j in range(self.num_clients):
                client_j = self.all_clients[j]
                if self.connected_graph[client_i.id][client_j.id]:
                    client_i.receive_neighbor_model(client_j.send_model())

        print("Aggregating model weights...")
        for client in self.participated_training_clients:
            client.aggregate()

    def evaluate_client(self):
        print("Evaluating model...")
        client_result_dic = {}
        for client in self.participated_training_clients:
            client_result_dic[client.id] = client.evaluate_model()

        metrics_keys = next(iter(client_result_dic.values())).keys()
        overall_results = {key: 0 for key in metrics_keys}

        for result in client_result_dic.values():
            for key in metrics_keys:
                overall_results[key] += result.get(key, 0)
        for key in overall_results.keys():
            overall_results[key] /= len(client_result_dic)
        return overall_results, client_result_dic


    def lr_scheduler(self):
        for client in self.participated_training_clients:
            client.update_lr()