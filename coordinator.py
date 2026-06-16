import math
import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method, Manager
from tqdm import tqdm

from utils import event_bus


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
        self.topology = getattr(args, 'topology', 'random')
        self.n_rounds = args.n_rounds
        self.wc_warm_mode = getattr(args, 'wc_warm_mode', 'neighbor')
        self.connected_graph = None
        self.current_lambda = None
        self._topo_dirty = True
        self.seed = args.seed
        self.n_job = args.n_job
        self._init_clients()
        if (self.device.type == 'cuda' or self.device.type == 'cpu') and self.n_job > 1:
            try:
                set_start_method('spawn')
            except RuntimeError as e:
                print("Start method 'spawn' already set or error setting it: ", str(e))

        self.interchange_model_method = self.interchange_model
        # 固定图（gossip=0）在启动时生成并完成全部成员状态的连通性核查（A5 前提）
        self.generate_connected_graph()
        self._refresh_topology_info()

    def _init_clients(self):
        print("Initializing initial clients...")
        pbar = tqdm(total=len(self.participated_training_clients))
        for client in self.participated_training_clients:
            client.set_init_model(deepcopy(self.init_model))
            client.init_client()
            pbar.update(1)
        pbar.clear()
        pbar.close()

    # ---------------- 拓扑：生成 / 连通性核查 / λ̂ ----------------

    def _active_ids(self):
        return [client.id for client in self.participated_training_clients]

    def _required_active_sets(self):
        """需要保证诱导子图连通的成员状态序列。

        固定图：Phase-1 在位集合及每个加入事件后的累积集合都必须连通
        （延迟客户端可能是割点，全图连通不蕴含诱导子图连通）。
        每轮随机图：只需当前活跃集合连通。
        """
        if self.gossip:
            return [frozenset(self._active_ids())]
        active = {c.id for c in self.all_clients if c.id not in self.client_delay_dict}
        states = [frozenset(active)]
        for cid, join_round in sorted(self.client_delay_dict.items(), key=lambda kv: kv[1]):
            if join_round >= self.n_rounds:
                continue  # 本次实验内不会加入
            active.add(cid)
            states.append(frozenset(active))
        return states

    @staticmethod
    def _is_connected(graph, ids):
        ids = list(ids)
        if len(ids) <= 1:
            return True
        id_set = set(ids)
        visited = {ids[0]}
        stack = [ids[0]]
        while stack:
            u = stack.pop()
            for v in id_set:
                if v not in visited and graph[u][v]:
                    visited.add(v)
                    stack.append(v)
        return len(visited) == len(id_set)

    def generate_connected_graph(self):
        if not self.gossip and self.connected_graph is not None:
            return

        if self.topology == 'ring':
            graph = [[0] * self.num_clients for _ in range(self.num_clients)]
            for i in range(self.num_clients):
                j = (i + 1) % self.num_clients
                graph[i][j] = graph[j][i] = 1
            self._check_or_raise(graph)
            self.connected_graph = graph
            self._topo_dirty = True
            return

        if self.topology == 'full':
            graph = [[1 if i != j else 0 for j in range(self.num_clients)]
                     for i in range(self.num_clients)]
            self.connected_graph = graph
            self._topo_dirty = True
            return

        # random 拓扑：重试直至全部成员状态的诱导子图连通（仅对称图执行核查）
        for _ in range(200):
            graph = self._build_random_graph()
            if self.symmetry == 0:
                break  # 旧有向模式：不在 A5 范围内，按原行为放行
            if all(self._is_connected(graph, s) for s in self._required_active_sets()):
                break
        else:
            raise RuntimeError(
                "200 次重试仍无法生成满足全部成员状态连通的对称图；"
                "请增大 num_conn 或检查延迟客户端配置")
        self.connected_graph = graph
        self._topo_dirty = True

    def _check_or_raise(self, graph):
        bad = [s for s in self._required_active_sets() if not self._is_connected(graph, s)]
        if bad:
            raise RuntimeError(
                f"拓扑 {self.topology} 在成员状态 {sorted(bad[0])} 上诱导子图不连通（A5 失效）；"
                "ring 拓扑请只配合单客户端延迟使用")

    def _random_regular_graph(self, n, d):
        """严格随机连通 d-正则图：每个节点恰好 d 个邻居、无自环/重边、连通。

        用 networkx 的 Steger–Wormald 算法生成（对大 d 仍高效，远优于配置模型的
        拒绝采样）；种子取自全局 random 状态以保证可复现且每次重试不同。内部重试
        直至全图连通（d>=3 几乎必然连通，d=2 为若干环需重试得到单环）。诱导子图
        （活跃集去掉延迟客户端）的连通性由 generate_connected_graph 外层重试保证。
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "对称拓扑（symmetry!=0）的严格 d-正则图生成需要 networkx，请先安装："
                "pip install networkx") from exc
        for _ in range(1000):
            seed = random.randrange(2 ** 31)
            try:
                g = nx.random_regular_graph(d, n, seed=seed)
            except nx.NetworkXError:
                continue  # 偶发生成失败（悬挂的最后若干 stub 无法配对），换种子重来
            if nx.is_connected(g):
                graph = [[0] * n for _ in range(n)]
                for u, v in g.edges():
                    graph[u][v] = graph[v][u] = 1
                return graph
        raise RuntimeError(f"1000 次重试仍无法生成连通的 {d}-正则图 (n={n})")

    def _build_random_graph(self):
        if self.symmetry != 0:
            # 参数检查：简单 d-正则图存在的充要条件是 d<=n-1 且 n*d 为偶
            if self.num_conn > self.num_clients - 1:
                raise ValueError("For undirected graphs, each node can connect up to num_clients - 1 node")
            if (self.num_clients * self.num_conn) % 2 != 0:
                raise ValueError("For undirected graphs, num_clients * num_conn must be even")
            return self._random_regular_graph(self.num_clients, self.num_conn)

        # 有向（symmetry=0）：旧版非对称图，不在 A5 范围内，仅保留兼容
        print("Generating a asymmetric connectivity diagram")
        graph = [[0 for _ in range(self.num_clients)] for _ in range(self.num_clients)]
        if self.num_conn >= self.num_clients:
            raise ValueError("For directed graphs, the out-of-out degree of each node "
                             "must be less than num_clients (self-looping is not allowed)")

        outdegree = [0] * self.num_clients

        nodes = list(range(self.num_clients))
        random.shuffle(nodes)
        for i in range(self.num_clients - 1):
            u = nodes[i]
            v = nodes[i+1]
            graph[u][v] = 1
            outdegree[u] += 1

        for u in range(self.num_clients):
            available_targets = [v for v in range(self.num_clients)
                                 if v != u and graph[u][v] == 0]
            random.shuffle(available_targets)

            needed = self.num_conn - outdegree[u]
            for v in available_targets[:needed]:
                graph[u][v] = 1
                outdegree[u] += 1
        return graph

    def _lambda_hat(self, active_ids, degrees):
        """活跃诱导子图上 lazy Metropolis 矩阵 W=(I+M)/2 的第二大特征值（研究模式精确值）。"""
        ids = sorted(active_ids)
        n = len(ids)
        if n <= 1:
            return 0.0
        index = {cid: i for i, cid in enumerate(ids)}
        M = np.zeros((n, n))
        for a in ids:
            for b in ids:
                if a < b and self.connected_graph[a][b]:
                    w = 1.0 / (1.0 + max(degrees[a], degrees[b]))
                    M[index[a]][index[b]] = M[index[b]][index[a]] = w
        for i in range(n):
            M[i][i] = 1.0 - M[i].sum()
        W = (np.eye(n) + M) / 2.0
        eigenvalues = np.linalg.eigvalsh(W)
        return float(eigenvalues[-2])

    def _refresh_topology_info(self):
        """向客户端下发当前活跃图的度数与精确 λ̂。

        等价于 WC 规范的 DEGREE 交换（§6.5 J4）与研究模式 λ̂ 特征值分解（§5.3）。
        加入事件发生时必须先于 set_init_model 调用，使新客户端拿到 λ̂_post。
        """
        if self.connected_graph is None or self.symmetry == 0:
            self.current_lambda = None
            self._topo_dirty = False
            return
        active = self._active_ids()
        active_set = set(active)
        degrees = {i: sum(self.connected_graph[i][j] for j in active_set if j != i)
                   for i in active_set}
        self.current_lambda = self._lambda_hat(active_set, degrees)
        for client in self.all_clients:
            if client is not None and hasattr(client, 'set_topology_info'):
                client.set_topology_info(degree=degrees.get(client.id, 0),
                                         lam=self.current_lambda,
                                         n_active=len(active_set))
        event_bus.emit('topology', lambda_hat=self.current_lambda, n_active=len(active_set))
        self._topo_dirty = False

    # ---------------- 评测指标（Algorithm 9 口径，float64） ----------------

    def _consensus_stats(self, clients):
        """返回 (θ̄ state_dict[float64/cpu], Ω)。Ω = 平均 ||θ_i − θ̄||²。"""
        if not clients:
            return None, None
        sds = [c.model.state_dict() for c in clients]
        keys = [k for k, v in sds[0].items() if v.is_floating_point()]
        mean = {}
        for k in keys:
            acc = sds[0][k].detach().cpu().double().clone()
            for sd in sds[1:]:
                acc += sd[k].detach().cpu().double()
            mean[k] = acc / len(sds)
        omega = 0.0
        for sd in sds:
            omega += sum(float(((sd[k].detach().cpu().double() - mean[k]) ** 2).sum())
                         for k in keys)
        return mean, omega / len(sds)

    @staticmethod
    def _distance_to_mean(client, mean):
        sd = client.model.state_dict()
        return math.sqrt(sum(float(((sd[k].detach().cpu().double() - mean[k]) ** 2).sum())
                             for k in mean))

    def consensus_error(self):
        _, omega = self._consensus_stats(self.participated_training_clients)
        return omega

    def stationarity_gradnorm2(self):
        """||∇f_n(θ̄)||²：全批本地数据、客户端等权平均、float64 聚合。"""
        clients = self.participated_training_clients
        mean_sd, _ = self._consensus_stats(clients)
        if mean_sd is None:
            return None
        scratch = deepcopy(self.init_model).to(self.device)
        target = scratch.state_dict()
        new_sd = {k: (mean_sd[k].to(dtype=v.dtype, device=v.device) if k in mean_sd else v)
                  for k, v in target.items()}
        scratch.load_state_dict(new_sd)
        scratch.eval()  # 确定性前向（dropout 关）；梯度照常回传
        params = [p for p in scratch.parameters() if p.requires_grad]
        grad_sum = None
        for client in clients:
            for p in params:
                p.grad = None
            n_i = 0
            for x, labels in client.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                loss = torch.nn.functional.cross_entropy(scratch(x), labels, reduction='sum')
                loss.backward()
                n_i += labels.numel()
            if n_i == 0:
                continue
            g = torch.cat([p.grad.detach().reshape(-1) for p in params]).cpu().double() / n_i
            grad_sum = g if grad_sum is None else grad_sum + g
        if grad_sum is None:
            return None
        grad_mean = grad_sum / len(clients)
        return float((grad_mean ** 2).sum())

    def current_eta(self):
        if not self.participated_training_clients:
            return None
        client = self.participated_training_clients[0]
        if client.optimizer is None:
            return None
        return float(client.optimizer.param_groups[0]['lr'])

    # ---------------- 训练 / 加入 / 交换 ----------------

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
        joining = []
        for client_id, delay_round in self.client_delay_dict.items():
            if delay_round != current_round:
                continue
            if client_id in participated_ids:
                print(f"Client {client_id} has started training before adding")
                continue
            if self.all_clients[client_id] is None:
                print(f"Warning: Client {client_id} not found in all_clients!")
                continue
            joining.append(client_id)
        if not joining:
            return []

        # 加入前快照：θ̄^{τ_k−} 与 Ω^{τ_k−}（R1 恒等式核对的左侧数据，V1/E1）
        incumbents = list(self.participated_training_clients)
        theta_bar_pre, omega_pre = self._consensus_stats(incumbents)

        for client_id in joining:
            self.participated_training_clients.append(self.all_clients[client_id])
            participated_ids.add(client_id)

        # λ̂_post 与新度数必须先于 set_init_model 生效（DEGREE 交换语义，§6.5）
        self._refresh_topology_info()

        for client_id in joining:
            client = self.all_clients[client_id]
            client.set_init_model(deepcopy(self.init_model))
            client.init_client()

        # 加入恒等式数据：D_k 与 Ω^{τ_k}（R1：Ω^{τ_k} = (n−1)/n·Ω^{τ_k−} + (n−1)/n²·D_k²）
        if theta_bar_pre is not None:
            d_k = {cid: self._distance_to_mean(self.all_clients[cid], theta_bar_pre)
                   for cid in joining}
            _, omega_post = self._consensus_stats(self.participated_training_clients)
            event_bus.emit('join_identity',
                           joining=joining,
                           omega_pre=omega_pre,
                           omega_post=omega_post,
                           D_k=d_k,
                           n_post=len(self.participated_training_clients))
        return joining

    def train_client(self, current_round):
        event_bus.set_round(current_round)
        new_clients = self._add_new_training_clients(current_round)
        if len(new_clients) != 0:
            print(f"New clients: {new_clients}")
        print("Training models...")
        self._clients_train()

    def _snapshot(self, payload):
        """发送内容快照：冻结张量，保证同步轮语义（先全部交换、后混合，式 (U) 次序）。

        修复原实现的引用别名问题：state_dict 引用在先聚合的客户端 load_state_dict
        时被原地改写，导致后聚合的客户端读到混合后的权重。
        """
        if torch.is_tensor(payload):
            return payload.detach().clone()
        if isinstance(payload, dict):
            return {k: self._snapshot(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple)):
            return type(payload)(self._snapshot(v) for v in payload)
        if isinstance(payload, (int, float, str, bool)) or payload is None:
            return payload
        return deepcopy(payload)

    def interchange_model(self, current_round):
        self.generate_connected_graph()
        if self._topo_dirty:
            self._refresh_topology_info()
        graph = self.connected_graph

        active_clients = [c for c in self.all_clients
                          if not (c.id in self.client_delay_dict
                                  and current_round < self.client_delay_dict[c.id])]
        active_ids = [c.id for c in active_clients]
        pre_add_ids = [cid for cid, r in self.client_delay_dict.items()
                       if r == current_round + 1 and cid not in set(active_ids)]

        # 每个发送方本轮只做一次快照，所有接收方共享同一份冻结副本
        sends = {c.id: self._snapshot(c.send_model()) for c in active_clients}

        for c in active_clients:
            for j in active_ids:
                if graph[c.id][j]:
                    c.receive_neighbor_model(sends[j])

        # pre_add（PULL#2，加入前一轮的新鲜快照）
        for cid in pre_add_ids:
            client = self.all_clients[cid]
            if self.wc_warm_mode == 'global_sim':
                sources = active_ids  # W-G 变体（规范 §4.2）：全局平均
            else:
                sources = [j for j in active_ids if graph[cid][j]]
            for j in sources:
                client.receive_neighbor_model(sends[j])

        print("Aggregating model weights...")
        for client in self.participated_training_clients:
            client.aggregate()

    def interchange_model_dfedpgp(self, current_round):
        self.generate_connected_graph()

        out_degree = [sum(row) for row in self.connected_graph]

        pre_add_clients = []
        for i in range(self.num_clients):
            if i in self.client_delay_dict and current_round + 1 == self.client_delay_dict[i]:
                pre_add_clients.append(i)

        for i in range(self.num_clients):
            if i in self.client_delay_dict and current_round < self.client_delay_dict[i]:
                continue

            sender = self.all_clients[i]
            u_i, mu_i = sender.send_model()
            deg_i = out_degree[i]

            if deg_i == 0:
                continue

            weight_ij = 1.0 / deg_i

            for j in range(self.num_clients):
                if not self.connected_graph[i][j]:
                    continue

                if j in self.client_delay_dict and current_round < self.client_delay_dict[j]:
                    continue

                receiver = self.all_clients[j]

                weighted_u = {k: v * weight_ij for k, v in u_i.items()}
                weighted_mu = mu_i * weight_ij
                receiver.receive_neighbor_model((weighted_u, weighted_mu))

        for i in pre_add_clients:
            receiver = self.all_clients[i]
            deg_i = out_degree[i] or 1
            w = 1.0 / deg_i

            for j in range(self.num_clients):
                if self.connected_graph[i][j]:
                    sender = self.all_clients[j]
                    if getattr(sender, "u", None) is None:
                        continue
                    u_j, mu_j = sender.send_model()
                    weighted_u = {k: v * w for k, v in u_j.items()}
                    weighted_mu = mu_j * w
                    receiver.receive_neighbor_model((weighted_u, weighted_mu))

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
