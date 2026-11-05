import random
import torch
from itertools import cycle
from utils.args import *
try:                                    # 仅旧的切分函数需要；dirfix* 用 numpy 自行划分
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None
from utils.utils import *
from torch.utils.data import Subset


def reduce_dataset(dataset, ratio):
    total_indices = list(range(len(dataset)))
    np.random.shuffle(total_indices)
    subset_indices = total_indices[:int(ratio * len(dataset))]
    return Subset(dataset, subset_indices), subset_indices


def split_data_with_dirichlet(num_clients, a, dataset, test_size, frac, seed):
    np.random.seed(seed)
    total_samples_per_label = {label: int(len(np.where(np.array(dataset.targets) == label)[0]) * frac)
                               for label in range(len(dataset.classes))}

    # 分配数据到客户端
    data_indices = [np.array([]) for _ in range(num_clients)]
    min_size = 0
    while min_size < 10:  # 确保每个客户端至少有10个样本
        # 为每个类别生成Dirichlet分布
        for label in range(len(dataset.classes)):
            label_indices = np.where(np.array(dataset.targets) == label)[0]
            np.random.shuffle(label_indices)
            label_indices = label_indices[:total_samples_per_label[label]]
            # 使用Dirichlet分布分配这个类别的索引
            distributed_indices = np.random.dirichlet(np.repeat(a, num_clients)) * len(label_indices)
            distributed_indices = np.cumsum(distributed_indices).astype(int)
            data_indices = [np.concatenate((data_indices[i], label_indices[
                                                             distributed_indices[i - 1]:distributed_indices[i]]))
                            if i != 0 else np.concatenate((data_indices[i], label_indices[:distributed_indices[i]]))
                            for i in range(num_clients)]
        min_size = min([len(i) for i in data_indices])
        data_indices = [i.astype(int) for i in data_indices]

        # 为每个客户端划分验证集
    train_val_split = {}
    for i in range(num_clients):
        train_idx, val_idx = train_test_split(data_indices[i], test_size=test_size, random_state=seed)
        train_val_split[i] = {'train': train_idx, 'val': val_idx}
    return train_val_split


def split_data_dirichlet_fixed_shard(num_clients, alpha, dataset, shard_size, test_size, seed):
    """每客户端固定 shard_size 个样本，标签比例 ~ Dir(alpha)（n-sweep 专用）。

    与 split_data_with_dirichlet 的区别：后者把每个标签池按 Dir 切给客户端，shard 大小
    因而不等；扫 n 时 shard 大小会随 n 变化，per-round 工作量与本地目标同时在动，
    n 的效应无法分离。此处固定 shard 大小，只让 n 变化。
    标签池抽空时从剩余最大的池子补齐，保证每个客户端恰好 shard_size 个样本。
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    n_labels = len(dataset.classes)
    pools = {c: list(rng.permutation(np.where(targets == c)[0])) for c in range(n_labels)}
    total = sum(len(v) for v in pools.values())
    if total < num_clients * shard_size:
        raise ValueError(f"data insufficient: need {num_clients * shard_size}, have {total}")

    train_val_split = {}
    for i in range(num_clients):
        proportions = rng.dirichlet(np.repeat(alpha, n_labels))
        counts = rng.multinomial(shard_size, proportions)
        picked = []
        for c in range(n_labels):
            take = min(int(counts[c]), len(pools[c]))
            for _ in range(take):
                picked.append(pools[c].pop())
        while len(picked) < shard_size:                       # label pool exhausted -> top up
            avail = [c for c in range(n_labels) if pools[c]]
            biggest = max(avail, key=lambda c: len(pools[c]))
            picked.append(pools[biggest].pop())
        idx = np.array(picked, dtype=int)
        # 用 numpy 做 train/val 划分（不引入 sklearn 依赖；FL 环境中未安装）
        rng.shuffle(idx)
        n_val = int(round(len(idx) * test_size))
        train_val_split[i] = {'train': idx[n_val:], 'val': idx[:n_val]}
    return train_val_split


def split_data_with_label(number_clients, number_client_label, dataset, test_size, frac, seed):
    np.random.seed(seed)

    number_labels = len(torch.unique(dataset.targets))

    if number_client_label > number_labels:
        raise ValueError("number_client_label cannot be greater than the total number of labels.")

    label_pool = np.tile(np.arange(number_labels), (number_client_label, 1)).T.flatten()
    np.random.shuffle(label_pool)

    label_to_clients = {i: [] for i in range(number_clients)}
    label_pool_cycle = cycle(label_pool)
    for i in range(number_clients):
        while len(label_to_clients[i]) < number_client_label:
            next_label = next(label_pool_cycle)
            if next_label not in label_to_clients[i]:
                label_to_clients[i].append(next_label)

    # 初始化为列表，每个客户端一个空列表
    clients_data_indices = [[] for _ in range(number_clients)]
    for label, clients in label_to_clients.items():
        label_indices = np.where(np.array(dataset.targets) == label)[0]
        np.random.shuffle(label_indices)  # 随机化索引
        frac_label_indices = label_indices[:int(len(label_indices) * frac)]  # 取 frac 比例的索引
        split_indices = np.array_split(frac_label_indices, len(clients))
        for i, client in enumerate(clients):
            clients_data_indices[client].extend(split_indices[i].tolist())

    # 转换每个客户端的列表为ndarray
    train_val_split = {}
    for i in range(len(clients_data_indices)):
        clients_data_indices[i] = np.array(clients_data_indices[i])
        train_idx, val_idx = train_test_split(clients_data_indices[i], test_size=test_size, random_state=seed)
        train_val_split[i] = {'train': train_idx, 'val': val_idx}

    return train_val_split


def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def split_dataset_by_clusters(n_clients, dataset, alpha, n_clusters, test_size, frac, seed):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param test_size: scale size of the test set
    :param seed:
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """

    n_classes = len(
        torch.unique(dataset.targets if torch.is_tensor(dataset.targets) else torch.tensor(dataset.targets)))
    if n_clusters == -1:
        n_clusters = n_classes

    rng = random.Random(seed)
    np.random.seed(seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        _, label = dataset[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster

    for cluster_id in range(n_clusters):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, idx in enumerate(cluster_split):
            clients_indices[client_id] += idx

    train_val_split = {}
    for i, idx in enumerate(clients_indices):
        train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=seed)
        train_val_split[i] = {'train': train_idx, 'val': val_idx}

    return train_val_split


def split_dataset_iid(n_clients, dataset, test_size, frac, seed):
    """
    Split dataset among `n_clients` in an IID fashion.

    :param n_clients: Number of clients to distribute the data to
    :param dataset: Dataset object with .targets for labels
    :param test_size: Proportion of the dataset to reserve for validation
    :param frac: Fraction of the dataset to use
    :param seed: Random seed for reproducibility
    :return: Dictionary with client indices mapping to {'train': indices, 'val': indices}
    """
    # Total number of samples to select
    total_samples = int(len(dataset) * frac)

    # Randomly select indices from the dataset
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    selected_indices = all_indices[:total_samples]

    # Split indices evenly among clients
    indices_per_client = np.array_split(selected_indices, n_clients)

    # For each client, split their indices into train and validation sets
    train_val_split = {}
    for i, indices in enumerate(indices_per_client):
        train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=seed)
        train_val_split[i] = {'train': train_indices, 'val': val_indices}

    return train_val_split


def save_client_indices(dir, dataset_name, split_method, indexes, alpha):
    os.makedirs(dir, exist_ok=True)
    if split_method != 'even':
        dataset_subdir = os.path.join(dir, dataset_name + "_" + split_method + "_" + str(alpha))
    else:
        dataset_subdir = os.path.join(dir, dataset_name + "_" + split_method)
    os.makedirs(dataset_subdir, exist_ok=True)

    for client_id, client_data in indexes.items():
        dataset_subdir_client = os.path.join(dataset_subdir, f"client_{client_id}")
        os.makedirs(dataset_subdir_client, exist_ok=True)
        train_file = os.path.join(dataset_subdir_client, "train_indexes.npy")
        val_file = os.path.join(dataset_subdir_client, f"val_indexes.npy")
        np.save(train_file, client_data['train'])
        np.save(val_file, client_data['val'])
        print(f"Saved training indices for client {client_id} at {train_file}")
        print(f"Saved validation indices for client {client_id} at {val_file}")


if __name__ == "__main__":
    args = parse_args_for_dataset()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    full_dataset = load_dataset(args.dataset_name)
    if args.split_method.startswith("dirfix"):
        indices = split_data_dirichlet_fixed_shard(args.clients_num, args.alpha, full_dataset,
                                                   args.shard_size, args.test_ratio, args.seed)
    elif args.split_method == "dirichlet":
        indices = split_data_with_dirichlet(args.clients_num, args.alpha, full_dataset, args.test_ratio, args.frac,
                                            args.seed)
    elif args.split_method == "label":
        indices = split_data_with_label(args.clients_num, args.number_label, full_dataset, args.test_ratio, args.frac,
                                        args.seed)
    elif args.split_method == "clusters":
        indices = split_dataset_by_clusters(args.clients_num, full_dataset, args.alpha, args.n_clusters,
                                            args.test_ratio, args.frac, args.seed)
    elif args.split_method == "even":
        indices = split_dataset_iid(args.clients_num, full_dataset, args.test_ratio, args.frac, args.seed)
    else:
        raise ValueError(f"split_method does not contain {args.split_method}")

    save_client_indices(args.dataset_indexes_dir, args.dataset_name, args.split_method, indices, args.alpha)
    print(f"Saved client indices to {args.dataset_indexes_dir}")
