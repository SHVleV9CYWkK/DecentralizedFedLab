import os
import random

import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST
from torchvision import transforms
from transformers import MobileBertForSequenceClassification
import torch.optim as optim
from models.cnn_model import CNNModel, LeafCNN1, LeNet, AlexNet, ResNet18, VGG16, ResNet50


def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'emnist':
        url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
        if url != EMNIST.url:
            print('The URL of the dataset is inconsistent with the latest URL')
            EMNIST.url = url
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))
             ]
        )
        dataset = EMNIST(root='./data', train=True, download=True, transform=transform, split="byclass")
    elif dataset_name == 'tiny_imagenet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
    elif dataset_name == 'mnist':
        transform = transforms.ToTensor()
        dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"dataset_name does not contain {dataset_name}")
    return dataset


def load_model(model_name, num_classes):
    if model_name == 'alexnet':
        model = AlexNet(num_classes)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes)
    elif model_name == 'vgg16':
        model = VGG16(num_classes)
    elif model_name == 'cnn':
        model = CNNModel(num_classes)
    elif model_name == 'leafcnn1':
        model = LeafCNN1(num_classes)
    elif model_name == 'lenet':
        model = LeNet(num_classes)
    elif model_name == 'mobilebart':
        model = MobileBertForSequenceClassification.from_pretrained("lordtt13/emo-mobilebert",
                                                                    num_labels=num_classes,
                                                                    ignore_mismatched_sizes=True)
        original_forward = model.forward

        def forward_with_logits_only(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            return outputs.logits

        model.forward = forward_with_logits_only
    else:
        raise ValueError(f"model_name does not contain {model_name}")
    return model


def get_client_data_indices(root_dir, dataset_name, split_method, alpha):
    dir_path = os.path.join(root_dir, f"{dataset_name}_{split_method}_{alpha}")

    if not os.path.exists(dir_path):
        dir_path = os.path.join(root_dir, f"{dataset_name}_{split_method}")
        if not os.path.exists(dir_path):
            raise ValueError(f"No matching dataset and split method found for {dataset_name} and {split_method}")

    # 获取客户端目录
    client_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    # 自动获取客户端数量
    num_clients = len(client_dirs)

    # 读取每个客户端的数据集索引
    client_indices = {}
    for client_dir in client_dirs:
        client_id = int(client_dir.split('_')[-1])
        client_indices[client_id] = {
            'train': os.path.join(dir_path, client_dir, 'train_indexes.npy'),
            'val': os.path.join(dir_path, client_dir, 'val_indexes.npy')
        }

    return client_indices, num_clients


def get_optimizer(optimizer_name, parameters, lr):
    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr)

    elif optimizer_name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9)

    elif optimizer_name == "adamw":
        return optim.AdamW(parameters, lr=lr)
    else:
        raise NotImplementedError(f"{optimizer_name} optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None, gated_learner=False):
    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        if gated_learner:
            # milestones = [n_rounds//2, 11*(n_rounds//12)]
            milestones = [3 * (n_rounds // 4)]
        else:
            milestones = [n_rounds // 2, 3 * (n_rounds // 4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif "reduce_on_plateau" in scheduler_name:
        last_word = scheduler_name.split("_")[-1]
        patience = int(last_word) if last_word.isdigit() else 10
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=0.75)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")


def get_client_delay_info(num_clients, delay_client_ratio, minimum_round, total_rounds, dist_type="single"):
    if total_rounds <= minimum_round:
        raise ValueError("total_rounds must be greater than minimum_round")
    if not (0 <= delay_client_ratio <= 1):
        raise ValueError("delay_client_ratio must be between 0 and 1")

    client_delay_info = {}
    client_ids = list(range(num_clients))

    if dist_type.lower() == "single":
        if num_clients < 1:
            raise ValueError("There must be at least one client to delay.")
        delayed_cid = random.choice(client_ids)
        join_round = minimum_round
        client_delay_info[delayed_cid] = join_round

    else:
        num_delayed = int(round(num_clients * delay_client_ratio))
        delayed_client_ids = set(random.sample(client_ids, num_delayed))

        if dist_type.lower() == "uniform":
            for cid in client_ids:
                if cid in delayed_client_ids:
                    join_round = random.randint(minimum_round + 1, total_rounds)
                    client_delay_info[cid] = join_round

        elif dist_type.lower() == "even":
            available_rounds = list(range(minimum_round+1, total_rounds+1))
            n_rounds = len(available_rounds)
            base = num_delayed // n_rounds
            remainder = num_delayed % n_rounds
            delayed_rounds_list = []
            for r in available_rounds:
                delayed_rounds_list.extend([r] * base)
            extra_rounds = random.sample(available_rounds, remainder) if remainder > 0 else []
            for r in extra_rounds:
                delayed_rounds_list.append(r)
            random.shuffle(delayed_rounds_list)

            for cid in client_ids:
                if cid in delayed_client_ids:
                    client_delay_info[cid] = delayed_rounds_list.pop() if delayed_rounds_list else minimum_round

        elif dist_type.lower() == "normal":
            mu = (minimum_round + 1 + total_rounds) / 2.0
            sigma = (total_rounds - minimum_round) / 4.0
            delayed_rounds = []
            for _ in range(num_delayed):
                while True:
                    r = int(round(random.gauss(mu, sigma)))
                    if minimum_round + 1 <= r <= total_rounds:
                        delayed_rounds.append(r)
                        break
            random.shuffle(delayed_rounds)
            for cid in client_ids:
                if cid in delayed_client_ids:
                    client_delay_info[cid] = delayed_rounds.pop() if delayed_rounds else minimum_round

        else:
            raise ValueError("Unidentified distribution type, use 'uniform', 'even', or 'normal'.")

    print(f"Delayed clients:{client_delay_info}")
    return client_delay_info


def save_log(eval_results, today_date, num_exper, current_rounds, args, client_id=None):
    today_dir = os.path.join(args.log_dir, today_date)
    os.makedirs(today_dir, exist_ok=True)

    dataset_dir = os.path.join(today_dir, args.dataset_name)
    os.makedirs(today_dir, exist_ok=True)

    method_name_dir = os.path.join(dataset_dir, args.fl_method)
    os.makedirs(method_name_dir, exist_ok=True)

    log_dir = os.path.join(method_name_dir, num_exper)
    os.makedirs(log_dir, exist_ok=True)

    if client_id is not None:
        client_dir = os.path.join(log_dir, "client_result")
        log_dir = os.path.join(client_dir, f"Client_{client_id}")
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = os.path.join(log_dir, "global_result")
        os.makedirs(log_dir, exist_ok=True)

    for metric, value in eval_results.items():
        file_path = os.path.join(log_dir, f"{metric}.csv")
        with open(file_path, 'a') as file:
            file.write(f"{current_rounds},{value}\n")


def get_experiment_num(today_date, args):
    method_name_dir = os.path.join(args.log_dir, today_date, args.dataset_name, args.fl_method)

    # 如果目录不存在，则返回 1 表示第一次实验
    if not os.path.exists(method_name_dir):
        return "1"

    # 列出 fl_type_dir 下所有的子目录，这些子目录名称应为数字，例如 "1", "2" 等
    exp_dirs = [
        d for d in os.listdir(method_name_dir)
        if os.path.isdir(os.path.join(method_name_dir, d)) and d.isdigit()
    ]

    if not exp_dirs:
        return "1"

    # 将实验目录名称转换为整数，取最大值后加 1 即为当前实验数
    exp_numbers = [int(d) for d in exp_dirs]
    next_exp_num = max(exp_numbers) + 1

    return str(next_exp_num)
