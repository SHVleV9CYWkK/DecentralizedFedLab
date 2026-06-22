#!/usr/bin/env python
"""实验矩阵启动器（对应实验清单 E1–E22 的 run 分组；幂等：已完成的 run 自动跳过）。

用法：
    python run_experiments.py --list                     # 列出全部实验组与 run 数
    python run_experiments.py G1_main_cifar10            # 跑一个组
    python run_experiments.py G2_grid G3_tau --dry_run   # 只打印命令不执行
    DFL_DEVICE=cuda python run_experiments.py G1_main_cifar10   # 环境变量覆盖设备

实验组 ↔ 实验清单对应：
    G1_main_*    E11/E16/E17/E18 主表（7 臂；同时供 E1/E2/E4/E10/E13 做纯分析）。
                 按数据集差异化（见 MAIN_SEEDS / MAIN_SCENARIOS）：
                   cifar10 / cifar100 : S1+S2 × 5 种子（统计骨架，完整显著性）= 各 70
                   emnist / tiny       : 仅 S1 × 3 种子（泛化广度）           = 各 21
    G2_grid      E6 校准质量 vs 事后网格（{η_c·2^-j}, j=0..6）
    G3_tau       E7 步长–迟到单调性（τ_k ∈ {0.2..0.9}T；E3 封顶阈值为其免费分析）
    G4_schedule  E9 调度无免费午餐（W + warmup/cosine/sqrt 衰减 vs 常数 η̂）
    G5_perturb   E5 R-W1 违规；E8 κ 扰动；E14 估计器保守方向；E4(c) W-G 全局平均
    G6_topology  E15 |N_k| / E20 谱隙（ring / full / random num_conn 扫描）
    G7_alpha     E19 异质性（Dirichlet α ∈ {0.1, 1.0}；α=0.4 复用 G1）

跑完后聚合：python -m analysis.aggregate
"""
import argparse
import json
import os
import subprocess
import sys

# ===================== 全局设备配置（按机器修改这一行） =====================
DEVICE = "cuda"            # Apple 笔记本: "mps"；GPU 服务器: "cuda"；默认: "cpu"
# 也可不改文件，用环境变量覆盖：DFL_DEVICE=cuda python run_experiments.py ...
DEVICE = os.environ.get("DFL_DEVICE", DEVICE)
# ==========================================================================

RESULTS_DIR = "results"
SEEDS = [42, 43, 44, 45, 46]      # 主表默认（E16：≥5 种子；同种子下各臂共享延迟抽取 → 配对比较）
SEEDS_SMALL = [42, 43, 44]        # 机制 / 敏感性实验

# 主表按数据集分配种子数。统计骨架（cifar10 / cifar100）用满 5 个（保证 Wilcoxon
# 显著性）；泛化广度（emnist / tiny）用 3 个。cifar100 墙钟实测比 emnist 快（50 客户端
# + GPU 高效 ResNet），故承担骨架；emnist（100 客户端，慢）降为广度。
# 日后补种子：在列表末尾追加（勿重排），幂等启动器只跑新增种子。
MAIN_SEEDS = {
    'cifar10': [42, 43, 44, 45, 46],
    'cifar100': [42, 43, 44, 45, 46],
    'emnist': [42, 43, 44],
    'tiny_imagenet': [42, 43, 44],
}

# 主表按数据集分配加入场景。骨架（cifar10 / cifar100）跑 S1+S2 做完整显著性；
# 广度（emnist / tiny）只跑 S1——S1 绑定全部理论与最干净的消融，S2（多客户端错峰
# = 现实性）由骨架数据集满配演示即可。消融 4 臂在所有数据集上完整保留。
MAIN_SCENARIOS = {
    'cifar10': ['S1', 'S2'],
    'cifar100': ['S1', 'S2'],
    'emnist': ['S1'],
    'tiny_imagenet': ['S1'],
}

# 全部 run 共享的地板参数（GR1：同地板比较；wc 客户端内部强制纯 SGD + 常数步长）
COMMON = {
    'optimizer_name': 'sgd',
    'scheduler_name': 'constant',
    'lr': 0.01,
    'batch_size': 64,
    'local_epochs': 1,
    'symmetry': 1,                # A5：对称混合矩阵
    'gossip': 0,                  # 固定图：λ̂ 精确可知（研究模式）
    'num_conn': 6,
    'n_job': 1,                   # WC 的 JOIN 公告板要求单进程
    'eval_every': 5,
    'split_method': 'dirichlet',
}

# 数据集 × 模型 × 轮数（客户端数由索引目录自动决定：cifar10/emnist=100、cifar100/tiny=50）
# 轮数须足够长，使 Phase-1 在 τ_k=0.5T 前进入平台（Δ̂_k 稳定）、且加入后窗口 T'=T−τ_k
# 仍长到脱离 cap-active——这是组件 C 的前提（短轮数下 V2/V4 不成立，见 2026-06 分析）。
DATASETS = {
    'cifar10': {'dataset_name': 'cifar10', 'model': 'lenet', 'alpha': 0.4, 'n_rounds': 200},
    'cifar100': {'dataset_name': 'cifar100', 'model': 'resnet18gn', 'alpha': 0.4, 'n_rounds': 200},
    'emnist': {'dataset_name': 'emnist', 'model': 'leafcnn1', 'alpha': 0.4, 'n_rounds': 150},
    'tiny_imagenet': {'dataset_name': 'tiny_imagenet', 'model': 'tinyvit', 'alpha': 0.4, 'n_rounds': 200},
}

# 实验臂：fl_method + WC 消融开关（E11 组件消融 + 三个 baseline）
ARMS = {
    'wc':      {'fl_method': 'wc'},
    'w_only':  {'fl_method': 'wc', 'wc_calibrate': 0},
    'c_only':  {'fl_method': 'wc', 'wc_warm_mode': 'cold'},
    'cold':    {'fl_method': 'wc', 'wc_warm_mode': 'cold', 'wc_calibrate': 0},
    'dfedavg': {'fl_method': 'dfedavg'},
    'dfedsam': {'fl_method': 'dfedsam'},
    'ellocal': {'fl_method': 'ellocal'},
}


def s1(n_rounds, frac=0.5):
    """S1：受控单事件加入（τ_k = frac·T；延迟客户端按种子随机抽取，臂间配对）。"""
    return {'temp_client_dist': 'single', 'set_single_delay_client': -1,
            'minimum_join_rounds': int(frac * n_rounds)}


def s2(n_rounds):
    """S2：错峰多事件加入（20% 客户端，τ ∈ (0.25T, T] 均匀抽取，臂间配对）。"""
    return {'temp_client_dist': 'uniform', 'delay_client_ratio': 0.2,
            'minimum_join_rounds': int(0.25 * n_rounds)}


def build_groups():
    groups = {}

    # ---- G1 主表：每数据集一组 ----
    for ds_key, ds in DATASETS.items():
        runs = []
        seeds = MAIN_SEEDS.get(ds_key, SEEDS)
        all_settings = {'S1': s1(ds['n_rounds']), 'S2': s2(ds['n_rounds'])}
        for setting_name in MAIN_SCENARIOS.get(ds_key, ['S1', 'S2']):
            setting = all_settings[setting_name]
            for arm_name, arm in ARMS.items():
                for seed in seeds:
                    runs.append((f"{ds_key}_{arm_name}_{setting_name}_s{seed}",
                                 {**COMMON, **ds, **setting, **arm, 'seed': seed}))
        groups[f'G1_main_{ds_key}'] = runs

    cifar10 = DATASETS['cifar10']
    base_s1 = {**COMMON, **cifar10, **s1(cifar10['n_rounds']), 'fl_method': 'wc'}

    # ---- G2 网格（E6）：η̂ := η_c·2^-j ----
    runs = []
    for j in range(7):
        for seed in SEEDS_SMALL:
            runs.append((f"grid_j{j}_s{seed}",
                         {**base_s1, 'wc_eta_frac': 0.5 ** j, 'seed': seed}))
    for seed in SEEDS_SMALL:   # 校准臂参照（与 G1 cifar10 wc S1 种子互补，亦可复用）
        runs.append((f"grid_calibrated_s{seed}", {**base_s1, 'seed': seed}))
    groups['G2_grid'] = runs

    # ---- G3 τ_k 扫描（E7/E3）----
    runs = []
    for frac in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for seed in SEEDS_SMALL:
            runs.append((f"tau{int(frac * 100)}_s{seed}",
                         {**COMMON, **cifar10, **s1(cifar10['n_rounds'], frac),
                          'fl_method': 'wc', 'seed': seed}))
    groups['G3_tau'] = runs

    # ---- G4 调度臂（E9）----
    runs = []
    for sched in ('sqrt_decay', 'cosine', 'warmup'):
        for seed in SEEDS_SMALL:
            runs.append((f"sched_{sched}_s{seed}",
                         {**base_s1, 'wc_post_schedule': sched, 'seed': seed}))
    groups['G4_schedule'] = runs

    # ---- G5 扰动 / 违规（E5、E8、E14、E4c）----
    runs = []
    for seed in SEEDS_SMALL:
        runs.append((f"fitted_s{seed}", {**base_s1, 'wc_warm_mode': 'fitted', 'seed': seed}))
        runs.append((f"globalsim_s{seed}", {**base_s1, 'wc_warm_mode': 'global_sim', 'seed': seed}))
    for kappa in (0.0625, 0.25, 4.0, 16.0):
        for seed in SEEDS_SMALL:
            runs.append((f"kappa{kappa}_s{seed}", {**base_s1, 'wc_kappa_g': kappa, 'seed': seed}))
    for c_l in (1.0, 4.0):
        for seed in SEEDS_SMALL:
            runs.append((f"cL{c_l}_s{seed}", {**base_s1, 'c_L': c_l, 'seed': seed}))
    for lam in (0.5, 0.99):
        for seed in SEEDS_SMALL:
            runs.append((f"lam{lam}_s{seed}", {**base_s1, 'lambda_hat_override': lam, 'seed': seed}))
    groups['G5_perturb'] = runs

    # ---- G6 拓扑 / 谱隙（E15、E20）----
    runs = []
    for topo in ('ring', 'full'):
        for seed in SEEDS_SMALL:
            runs.append((f"{topo}_s{seed}", {**base_s1, 'topology': topo, 'seed': seed}))
    for conn in (4, 10):
        for seed in SEEDS_SMALL:
            runs.append((f"conn{conn}_s{seed}", {**base_s1, 'num_conn': conn, 'seed': seed}))
    groups['G6_topology'] = runs

    # ---- G7 异质性（E19）----
    runs = []
    for alpha in (0.1, 1.0):
        for arm_name in ('wc', 'w_only', 'dfedavg'):
            for seed in SEEDS:
                runs.append((f"a{alpha}_{arm_name}_s{seed}",
                             {**COMMON, **cifar10, **s1(cifar10['n_rounds']),
                              **ARMS[arm_name], 'alpha': alpha, 'seed': seed}))
    groups['G7_alpha'] = runs

    return groups


def is_completed(group, run_name):
    path = os.path.join(RESULTS_DIR, group, run_name, 'summary.json')
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f).get('status') == 'COMPLETED'
    except Exception:
        return False


def build_command(group, run_name, config):
    cmd = [sys.executable, 'main.py',
           '--device', DEVICE,
           '--results_dir', RESULTS_DIR,
           '--exp_group', group,
           '--run_name', run_name]
    for key, value in config.items():
        cmd += [f'--{key}', str(value)]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('groups', nargs='*', help='要运行的实验组（见 --list）')
    parser.add_argument('--list', action='store_true', help='列出全部实验组')
    parser.add_argument('--dry_run', action='store_true', help='只打印命令不执行')
    args = parser.parse_args()

    all_groups = build_groups()
    if args.list or not args.groups:
        print(f"device = {DEVICE}\n")
        for name, runs in all_groups.items():
            done = sum(1 for run_name, _ in runs if is_completed(name, run_name))
            print(f"  {name:24s} {done}/{len(runs)} 完成")
        return

    failed = []
    for group in args.groups:
        if group not in all_groups:
            print(f"未知实验组: {group}（--list 查看全部）")
            continue
        runs = all_groups[group]
        for i, (run_name, config) in enumerate(runs):
            if is_completed(group, run_name):
                print(f"[skip {i + 1}/{len(runs)}] {group}/{run_name}")
                continue
            cmd = build_command(group, run_name, config)
            print(f"[run  {i + 1}/{len(runs)}] {group}/{run_name}  (device={DEVICE})")
            if args.dry_run:
                print('   ', ' '.join(cmd))
                continue
            result = subprocess.run(cmd)
            if result.returncode != 0:
                failed.append(f"{group}/{run_name}")
                print(f"[FAIL] {group}/{run_name} (exit {result.returncode})，继续下一个")

    if failed:
        print(f"\n失败 {len(failed)} 个 run：")
        for name in failed:
            print(f"  {name}")
    else:
        print("\n全部完成。聚合：python -m analysis.aggregate")


if __name__ == '__main__':
    main()
