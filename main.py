import random
import time
from datetime import datetime

import numpy as np
import torch
import logging
logging.getLogger().setLevel(logging.ERROR)

from clients.client_factory import create_client
from coordinator import Coordinator
from utils import event_bus
from utils.args import parse_args
from utils.run_logger import RunLogger
from utils.utils import load_model, load_dataset, get_client_data_indices, get_client_delay_info


def execute_fed_process(coordinator, args, run_logger):
    join_rounds = sorted({r for r in coordinator.client_delay_dict.values() if r < args.n_rounds})
    gradnorm_samples = []
    final_overall = {}

    for r in range(args.n_rounds):
        print(f"Round {r}")
        start_time = time.time()
        coordinator.train_client(r)
        coordinator.interchange_model_method(r)
        overall_results, client_results = coordinator.evaluate_client()

        # 网络级指标（Algorithm 9 口径）：Ω 每轮；||∇f_n(θ̄)||² 按 eval_every 采样，
        # 并在加入轮邻域加密采样（窗口平稳性 M 的数据来源）
        network_metrics = {
            'Omega': coordinator.consensus_error(),
            'eta': coordinator.current_eta(),
            'lambda_hat': coordinator.current_lambda,
            'n_active': len(coordinator.participated_training_clients),
        }
        need_gradnorm = args.eval_every > 0 and (
            r % args.eval_every == 0
            or r == args.n_rounds - 1
            or r in join_rounds or (r - 1) in join_rounds or (r + 1) in join_rounds
        )
        if need_gradnorm:
            network_metrics['gradnorm2'] = coordinator.stationarity_gradnorm2()
            gradnorm_samples.append((r, network_metrics['gradnorm2']))

        run_logger.log_metrics(r, 'overall', overall_results)
        for client_id, client_result in client_results.items():
            run_logger.log_metrics(r, 'client', client_result, client_id=client_id)
        run_logger.log_metrics(r, 'network', network_metrics)

        end_time = time.time()
        eval_results_str = ', '.join(
            [f"{metric.capitalize()}: {value:.4f}" for metric, value in overall_results.items()])
        print(f"Training time: {(end_time - start_time):.2f}. Evaluation Results: {eval_results_str}")
        print(f"Omega: {network_metrics['Omega']:.6g}, eta: {network_metrics['eta']}")

        accuracies = [res["accuracy"] for res in client_results.values()]
        max_cid = max(client_results, key=lambda cid: client_results[cid]["accuracy"])
        min_cid = min(client_results, key=lambda cid: client_results[cid]["accuracy"])
        print(f"Client Accuracy — Max: {max(accuracies):.4f} (Client {max_cid}), "
              f"Min: {min(accuracies):.4f} (Client {min_cid})")

        coordinator.lr_scheduler()
        final_overall = overall_results
        print(f"------------")

    # —— summary：窗口平均平稳性 M（GR2 认证指标，采样近似）——
    summary = {'final': final_overall, 'n_rounds': args.n_rounds, 'join_rounds': join_rounds}
    for tau in join_rounds:
        values = [g for (r, g) in gradnorm_samples if r >= tau and g is not None]
        if values:
            summary[f'M_window_tau{tau}'] = sum(values) / len(values)
    if join_rounds:
        first = f'M_window_tau{join_rounds[0]}'
        if first in summary:
            summary['M_window'] = summary[first]
    return summary


def execute_experiment(args, device):
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    if RunLogger.is_completed(args.results_dir, args.exp_group, run_name):
        print(f"[skip] {args.exp_group}/{run_name} 已完成（幂等跳过）")
        return

    run_logger = RunLogger(args.results_dir, args.exp_group, run_name, vars(args),
                           device=str(device))
    event_bus.attach(run_logger)
    status = 'INTERRUPTED'
    summary = {}
    t_start = time.time()
    try:
        full_dataset = load_dataset(args.dataset_name)
        model = load_model(args.model, num_classes=len(full_dataset.classes)).to(device)
        client_indices, num_clients = get_client_data_indices(
            args.dataset_indexes_dir, args.dataset_name, args.split_method, args.alpha)

        clients = create_client(num_clients, args, client_indices, full_dataset, device)

        client_delay = get_client_delay_info(
            num_clients, args.delay_client_ratio, args.minimum_join_rounds, args.n_rounds,
            args.temp_client_dist, args.set_single_delay_client)
        run_logger.save_config('client_delay', {str(k): v for k, v in client_delay.items()})

        coordinator = Coordinator(clients, model, device, client_delay, args)
        run_logger.save_config('lambda_hat_initial', coordinator.current_lambda)

        if args.fl_method == "dfedpgp":
            coordinator.interchange_model_method = coordinator.interchange_model_dfedpgp

        summary = execute_fed_process(coordinator, args, run_logger)
        status = 'COMPLETED'
    finally:
        summary['wall_time_sec'] = time.time() - t_start
        run_logger.finalize(summary, status)
        event_bus.detach()
        print(f"[{status}] {args.exp_group}/{run_name}")


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

    execute_experiment(arguments, compute_device)
    print("Done")

if __name__ == '__main__':
    main()
