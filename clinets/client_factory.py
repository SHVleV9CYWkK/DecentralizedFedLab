from clinets.dfl_method_clients.dfedavg_client import DFedAvgClient
from clinets.dfl_method_clients.dfedcad_client import DFedCADClient
from clinets.dfl_method_clients.dfedmtkd_client import DFedMTKDClient
from clinets.dfl_method_clients.dfedmtkdrl_client import DFedMTKDRLClient
from clinets.dfl_method_clients.dfedpgp import DFedPGPClient


def create_client(num_client, args, dataset_index, full_dataset, device):
    train_hyperparam = {
        'optimizer_name': args.optimizer_name,
        'lr': args.lr,
        'bz': args.batch_size,
        'local_epochs': args.local_epochs,
        'n_rounds': args.n_rounds,
        'scheduler_name': args.scheduler_name
    }

    fl_type = args.fl_method
    if fl_type == "dfedavg":
        client_class = DFedAvgClient
    elif "dfedcad" == fl_type:
        client_class = DFedCADClient
        train_hyperparam['lambda_kd'] = args.lambda_kd
        train_hyperparam['n_clusters'] = args.n_clusters
        train_hyperparam['lambda_alignment'] = args.lambda_alignment
        train_hyperparam['base_decay_rate'] = args.base_decay_rate
    elif "dfedmtkdrl" == fl_type:
        client_class = DFedMTKDRLClient
        train_hyperparam['lambda_kd'] = args.lambda_kd
    elif "dfedmtkd" == fl_type:
        client_class = DFedMTKDClient
        train_hyperparam['lambda_kd'] = args.lambda_kd
    elif "dfedpgp" == fl_type:
        client_class = DFedPGPClient

    else:
        raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

    clients_list = [None] * num_client
    for idx in range(num_client):
        clients_list[idx] = client_class(idx, dataset_index[idx], full_dataset, train_hyperparam, device)

    return clients_list
