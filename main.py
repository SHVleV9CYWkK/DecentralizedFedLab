import random
import time
from datetime import datetime
import numpy as np
import torch
import logging
logging.getLogger().setLevel(logging.ERROR)

from clinets.client_factory import create_client
from coordinator import Coordinator
from utils.args import parse_args
from utils.experiment_logger import ExperimentLogger
from utils.utils import load_model, load_dataset, get_client_data_indices, \
    get_client_delay_info, save_log, get_experiment_num


def execute_fed_process(coordinator, args, today_date, exper_num):
    for r in range(args.n_rounds):
        print(f"Round {r}")
        start_time = time.time()
        coordinator.train_client(r)
        coordinator.interchange_model(r)
        overall_results, client_results = coordinator.evaluate_client()
        end_time = time.time()

        eval_results_str = ', '.join([f"{metric.capitalize()}: {value:.4f}" for metric, value in overall_results.items()])
        print(f"Training time: {(end_time - start_time):.2f}. Evaluation Results: {eval_results_str}")

        accuracies = [res["accuracy"] for res in client_results.values()]
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        max_cid = max(client_results, key=lambda cid: client_results[cid]["accuracy"])
        min_cid = min(client_results, key=lambda cid: client_results[cid]["accuracy"])
        print(f"Client Accuracy — Max: {max_acc:.4f} (Client {max_cid}), Min: {min_acc:.4f} (Client {min_cid})")

        save_log(overall_results, today_date, exper_num, r, args)
        for client_id, client_result in client_results.items():
            save_log(client_result, today_date, exper_num, r, args, client_id)
        coordinator.lr_scheduler()
        print(f"------------")


def execute_experiment(args, device, exper_num, today_date, logger):
    full_dataset = load_dataset(args.dataset_name)
    model = load_model(args.model, num_classes=len(full_dataset.classes)).to(device)
    client_indices, num_clients = get_client_data_indices(args.dataset_indexes_dir, args.dataset_name,
                                                          args.split_method, args.alpha)

    clients = create_client(num_clients, args, client_indices, full_dataset, device)

    client_delay = get_client_delay_info(num_clients, args.delay_client_ratio, args.minimum_join_rounds, args.n_rounds,
                                         args.temp_client_dist)

    logger.save("client_delay", client_delay)

    coordinator = Coordinator(clients, model, device, client_delay, args)

    execute_fed_process(coordinator, args, today_date, exper_num)


def main():
    arguments = parse_args()

    torch.manual_seed(arguments.seed)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)

    if arguments.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(arguments.seed)
        compute_device = torch.device("cuda")
    elif arguments.device == "mps" and torch.backends.mps.is_available():
        compute_device = torch.device("mps:0")
    else:
        compute_device = torch.device("cpu")
    print(f"Using device: {compute_device}")

    today_date = datetime.today().strftime('%Y-%m-%d')

    experiment_num = get_experiment_num(today_date, arguments)

    with ExperimentLogger(today_date, experiment_num, compute_device, arguments) as logger:
        execute_experiment(arguments, compute_device, experiment_num, today_date, logger)
        print("Done")

if __name__ == '__main__':
    main()
