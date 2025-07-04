from clients.dfl_method_clients.dfedavg_client import DFedAvgClient
from clients.dfl_method_clients.dfedcad_client import DFedCADClient
from clients.dfl_method_clients.dfedmtkd_client import DFedMTKDClient
from clients.dfl_method_clients.dfedmtkdrl_client import DFedMTKDRLClient
from clients.dfl_method_clients.dfedpgp_clent import DFedPGPClient
from clients.dfl_method_clients.dfedsam_client import DFedSAMClient
from clients.dfl_method_clients.fedgo_client import FedGOClient
from clients.dfl_method_clients.qfedcg_client import QFedCGClient
from clients.dfl_method_clients.retfhd_client import ReTFHDClient


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
    elif "dfedcad" in fl_type:
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
    elif "dfedsam" == fl_type:
        client_class = DFedSAMClient
        train_hyperparam['rho'] = args.rho
    elif "fedgo" == fl_type:
        client_class = FedGOClient
        train_hyperparam['lambda_kd'] = args.lambda_kd
    elif "qfedcg" == fl_type:
        client_class = QFedCGClient
    elif "retfhd" == fl_type:
        client_class = ReTFHDClient

    else:
        raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

    clients_list = [None] * num_client
    for idx in range(num_client):
        clients_list[idx] = client_class(idx, dataset_index[idx], full_dataset, train_hyperparam, device)

    return clients_list
