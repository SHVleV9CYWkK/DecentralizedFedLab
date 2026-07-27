import argparse


def parse_args_for_dataset():
    parser = argparse.ArgumentParser(description="Dataset splitting for federated learning")
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'emnist', 'mnist', 'yahooanswers', 'tiny_imagenet'],
                        help='dataset name')
    parser.add_argument('--clients_num', type=int, default=10, help='number of clients')
    parser.add_argument('--n_clusters', type=int, default=-1, help='number of clusters using clusters split method')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--split_method', type=str, default='train', choices=['dirichlet', 'label', 'clusters', 'even'],
                        help='The methods of splitting the data set to generate non-IID are dirichlet and label '
                             'respectively. dirichlet is using dirichlet distributed. label indicates that the client '
                             'owns a subset of label')
    parser.add_argument(
        '--alpha', type=float, default=0.4,
        help='Parameters that control the degree of non-IID.'
             'The smaller the alpha, the greater the task difference',
    )

    parser.add_argument(
        '--frac', type=float, default=1.0,
        help='The proportion of a partial dataset to the entire dataset is adopted')

    parser.add_argument(
        '--test_ratio', type=float, default=0.2,
        help='The proportion of the test set to the overall dataset')

    parser.add_argument(
        '--number_label', type=int, default=2,
        help='Parameters that control the degree of non-IID.'
             'Controls the number of label types owned by the local client with label split method',
    )
    parser.add_argument('--dataset_indexes_dir', type=str, default='client_indices',
                        help='The root directory of the local client dataset index')

    args = parser.parse_args()
    return args


def parse_args_for_visualization():
    parser = argparse.ArgumentParser(description="Visualize the parameters of the training process")
    parser.add_argument('--log_dir', type=str, required=True, help='log directory')
    parser.add_argument('--save_dir', type=str, default=None, help='save directory')
    args = parser.parse_args()
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fl_method', type=str, default='dfedavg', choices=['dfedavg', 'dfedcad', 'dfedmtkd', 'dfedmtkdrl', 'dfedpgp', 'dfedsam', 'fedgo', 'qfedcg', 'retfhd', 'ellocal', 'localonly', 'wc'], help='Decentralized federated learning method')
    parser.add_argument('--dataset_name', type=str, default='emnist', choices=['cifar10', 'cifar100', 'emnist', 'mnist', 'tiny_imagenet'],
                        help='dataset name')
    parser.add_argument('--alpha', type=float, default=0.4, help='The alpha of the dataset, which is used to select the dataset')
    parser.add_argument('--model', type=str, default='lenet', choices=['cnn', 'alexnet', 'leafcnn1', 'lenet', 'mobilebart', 'resnet18', 'resnet18gn', 'vgg16', 'resnet50', 'tinyvit'],
                        help='model name')
    parser.add_argument('--optimizer_name', type=str, default='adam', choices=['sgd', 'adam', 'adamw'],
                        help='The name of the optimizer used')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate of the local client during training')
    parser.add_argument('--local_epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_rounds', type=int, default=50, help='number of global rounds')
    parser.add_argument('--scheduler_name', type=str, default='reduce_on_plateau', choices=['sqrt', 'linear', 'constant', 'cosine_annealing', 'multi_step', 'reduce_on_plateau'],
                        help='Select the name of the learning rate scheduler')
    parser.add_argument('--num_conn', type=int, default=10, help='The number of neighbors connected to by the client')
    parser.add_argument('--symmetry', type=int, default=0, help='Symmetry of the connected graph, non-0 represents symmetry')
    parser.add_argument('--gossip', type=int, default=1, help='Gossip protocol status identification, non-0 means enabling gossip protocol')
    parser.add_argument('--delay_client_ratio', type=float, default=0.5, help='The proportion of delayed client joining')
    parser.add_argument('--set_single_delay_client', type=int, default=-1, help='temp_client_dist When set to single, a specific client id is formulated. If -1, then randomly')
    parser.add_argument('--n_clusters', type=int, default=16, help='The number of weight clusters used by DKM')
    parser.add_argument('--base_decay_rate', type=float, default=0.5, help='Momentum updates the base attenuation rate locally')
    parser.add_argument('--minimum_join_rounds', type=int, default=25, help='Number of rounds to start joining a new client')
    parser.add_argument('--temp_client_dist', type=str, default='single', choices=['uniform', 'even', 'normal', 'single'], help='Temporarily join the client distribution')
    parser.add_argument('--lambda_kd', type=float, default=0.1, help='Distillation strength')
    parser.add_argument('--lambda_alignment', type=float, default=0.1, help='Alignment strength')
    parser.add_argument('--lambda_feature_kd', type=float, default=0.1, help='Distillation strength acting on feature parts')
    parser.add_argument('--n_job', type=int, default=1, help='The number of processes that execute client training in parallel in the server')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker processes. Default 0; auto-set to 4 for tiny_imagenet (ImageFolder/JPEG decode). Explicit >0 overrides for all datasets. Requires n_job=1.')
    parser.add_argument('--augment', type=int, default=0, help='1 = standard train-time augmentation (RandomCrop pad=4 + horizontal flip) on the training loader only; eval/E_k/full-batch metrics stay clean')
    parser.add_argument('--rho', type=float, default=0.05, help='Sharpness-Aware Minimization radius of DFedSAM')
    parser.add_argument('--lambda_hat', type=float, default=0.9, help='WC method: conservative upper bound of the post-join mixing-matrix spectral parameter lambda_n (fallback when no exact value is available; overestimate = safe)')
    parser.add_argument('--eta_min_frac', type=float, default=0.0, help='WC method: optional guardrail, eta_hat >= eta_min_frac * eta_c (0 = off, spec default)')
    parser.add_argument('--zeta2', type=float, default=1.0, help='WC method: heterogeneity constant zeta^2 used only by the near-noiseless calibration branch')
    # —— 实验记录（results/<exp_group>/<run_name>/）——
    parser.add_argument('--exp_group', type=str, default='adhoc', help='Experiment group name (directory under results_dir)')
    parser.add_argument('--run_name', type=str, default='', help='Run name; empty = timestamp. Completed runs are skipped (idempotent)')
    parser.add_argument('--results_dir', type=str, default='results', help='Root directory of structured run logs')
    parser.add_argument('--eval_every', type=int, default=5, help='Rounds between stationarity gradient-norm evaluations (<=0 disables)')
    # —— 拓扑 ——
    parser.add_argument('--topology', type=str, default='random', choices=['random', 'ring', 'full'], help='Communication topology family (symmetric graphs only for ring/full)')
    # —— WC 消融开关（实验清单 E4/E5/E6/E8/E9/E14）——
    parser.add_argument('--wc_warm_mode', type=str, default='neighbor', choices=['neighbor', 'global_sim', 'cold', 'fitted'], help='WC warm start: neighbor=W-N (default), global_sim=W-G, cold=no warm start, fitted=deliberate R-W1 violation (E5)')
    parser.add_argument('--wc_calibrate', type=int, default=1, help='WC: 1 = calibrate eta_hat at join (component C), 0 = keep default lr (W-only arm)')
    parser.add_argument('--wc_post_schedule', type=str, default='constant', choices=['constant', 'sqrt_decay', 'cosine', 'warmup'], help='WC: post-join lr schedule shape (E9 ablation; spec default constant)')
    parser.add_argument('--wc_eta_frac', type=float, default=0.0, help='WC: if >0, force eta_hat = frac * eta_c (E6 grid {eta_c*2^-j})')
    parser.add_argument('--wc_kappa_g', type=float, default=1.0, help='WC: multiplicative perturbation on G_n estimate (E8 miscalibration robustness)')
    parser.add_argument('--c_L', type=float, default=2.0, help='WC: conservative factor on the L estimate (must be >=1; E14 sweeps it)')
    parser.add_argument('--lambda_hat_override', type=float, default=-1.0, help='WC: if >0, force lambda_hat to this value, bypassing the exact eigenvalue (E14)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='determine the computing platform')
    parser.add_argument('--split_method', type=str, choices=['dirichlet', 'label', 'clusters', 'even'],
                        help='The methods of splitting the data set to generate non-IID are dirichlet and label '
                             'respectively. dirichlet is using dirichlet distributed. label indicates that the client '
                             'owns a subset of label')
    parser.add_argument('--dataset_indexes_dir', type=str, default='client_indices',
                        help='The root directory of the local client dataset index')

    args = parser.parse_args()

    if args.fl_method == 'dfedcad' and args.lambda_alignment == 0.0:
        args.fl_method = 'dfedcad_without_alignment'
    return args
