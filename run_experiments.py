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
    G8_calib_probe  组件 C 诊断：C 表现差是否因 cap-active（稀疏图→η_c 太小）。
                 隔离 C（wc vs w_only）× 连通度(conn6/full) × τ_k(0.5/0.9T) × c_L(2/1)
    G9_grid_full    V2 off-cap：full 拓扑 η 网格 vs 校准（conn6 版被 cap 污染）
    G10_tau_full    V4 阶梯 off-cap：full 拓扑 τ_k∈{0.2..0.9}T 全扫描
    G11_kappa_full  E8/R5 off-cap：κ 鲁棒性（capped 下 κ 无效，G5 κ 臂作废）
    G12_sched_full  E9 off-cap：调度 vs 常数 η̂
    G13_lrgrid_*    主表 v2 前置：cifar100/tiny 的地板 lr 网格（带增强）
    G14_main_aug_*  主表 v2：调好的 lr* + 增强 + 预训练卷积初始化，6 臂
                    （先跑 G13，把 lr* 填入 LR_STAR 后本组才生成）

注意：全部 SGD 无动量（get_optimizer momentum=0，同地板 GR1）；改动momentum前
跑的 baseline 结果 (m=0.9) 与新口径不同环境，须删除对应 run 目录后重跑。

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

# ============ 主表 v2 的调好地板 lr ============
# None = 从 G13 网格的落盘结果自动解析（见 resolve_lr_star：按平均最终准确率选取，
# 要求 4 档 lr × 全部种子均 COMPLETED 才生效，防止半程数据误选）。
# 手动填入数值可覆盖自动解析，例如 {'cifar100': 0.003, ...}。
LR_STAR = {
    'cifar100': None,
    'tiny_imagenet': None,
}

GRID_LRS = (0.03, 0.01, 0.003, 0.001)


def resolve_lr_star(ds_key):
    """自动解析主表 v2 的地板 lr*：LR_STAR 手动值优先；否则读 G13 网格结果。

    选取标准（论文须声明）：固定 lr 臂按其最优代表出场 —— lr* = 各档 lr 的
    平均最终准确率最大者。网格不完整（缺档或缺种子）时返回 None，G14 不生成。
    """
    if LR_STAR.get(ds_key) is not None:
        return LR_STAR[ds_key]
    group_dir = os.path.join(RESULTS_DIR, f'G13_lrgrid_{ds_key}')
    if not os.path.isdir(group_dir):
        return None
    acc_by_lr = {}
    for run_name in os.listdir(group_dir):
        run_dir = os.path.join(group_dir, run_name)
        try:
            with open(os.path.join(run_dir, 'summary.json'), encoding='utf-8') as f:
                summary = json.load(f)
            with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get('status') != 'COMPLETED':
            continue
        acc = summary.get('final', {}).get('accuracy')
        if acc is not None:
            acc_by_lr.setdefault(config['args']['lr'], []).append(acc)
    if set(acc_by_lr) != set(GRID_LRS) or \
            any(len(v) < len(SEEDS_SMALL) for v in acc_by_lr.values()):
        return None
    means = {lr: sum(v) / len(v) for lr, v in acc_by_lr.items()}
    lr_star = max(means, key=means.get)
    table = '  '.join(f'lr={lr}:{means[lr]:.4f}' for lr in sorted(means, reverse=True))
    print(f"[auto] {ds_key} 网格 → lr* = {lr_star}   ({table})")
    return lr_star

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
    'tiny_imagenet': {'dataset_name': 'tiny_imagenet', 'model': 'resnet18gn', 'alpha': 0.4, 'n_rounds': 200},
}

# 实验臂：fl_method + WC 消融开关（E11 组件消融 + baseline）
# 论文标注：cold ≡ D-PSGD (Lian et al. 2017)；w_only ≡ D-PSGD + aggregate-on-join。
# dfedsam 已移除：SAM 依赖动量，与无动量同地板（GR1）不兼容，三数据集均退化至
# 随机水平；论文 Related Work 提及即可。
ARMS = {
    'wc':      {'fl_method': 'wc'},
    'w_only':  {'fl_method': 'wc', 'wc_calibrate': 0},
    'c_only':  {'fl_method': 'wc', 'wc_warm_mode': 'cold'},
    'cold':    {'fl_method': 'wc', 'wc_warm_mode': 'cold', 'wc_calibrate': 0},
    'dfedavg': {'fl_method': 'dfedavg'},
    'ellocal': {'fl_method': 'ellocal'},
    # 不协作下界：完全不参与混合（动机检验——加入网络 vs 自己单练）。
    # 注意其 Ω/M 语义特殊：各客户端轨迹独立，共识误差与"全网平均模型的梯度"
    # 不再刻画一个协作系统，论文中按参考线报告 acc、不与协作臂比 M。
    'localonly': {'fl_method': 'localonly'},
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
        for arm_name in ('wc', 'w_only', 'dfedavg', 'ellocal'):
            for seed in SEEDS:
                runs.append((f"a{alpha}_{arm_name}_s{seed}",
                             {**COMMON, **cifar10, **s1(cifar10['n_rounds']),
                              **ARMS[arm_name], 'alpha': alpha, 'seed': seed}))
    groups['G7_alpha'] = runs

    # ---- G8 校准探针：确认"C 表现差是否因 cap-active（稀疏图→η_c 太小）"----
    # cifar10 长轮（已平台）；只隔离 C（wc vs w_only，二者同为 warm，仅校准 on/off）。
    # 三个杠杆让 η̂ 脱离 cap，看 C 是否随之"由害转益"、V4 是否转为递增：
    #   连通度：conn6(λ≈0.87,capped 基线) vs full(λ=0.5,off-cap)
    #   加入点：τ_k=0.5T（长窗口）vs 0.9T（短窗口=obs.5 的战场）
    #   c_L   ：2(默认) vs 1（更小 L̂ → 更大 η_c，最省的抬 η_c 手段）
    # 读法：① full/off-cap 或 c_L=1 下 wc 是否追平/反超 w_only（C 由害转益）
    #       ② τ_k 0.5→0.9 时 η̂ 是否递增（V4）③ off-cap 后 η̂ 是否落进 √ 分支
    runs = []
    c10 = {**COMMON, **DATASETS['cifar10']}            # n_rounds=200, lr=0.01, 已平台
    T10 = DATASETS['cifar10']['n_rounds']
    conn = {
        'conn6': {'topology': 'random', 'num_conn': 6},   # 稀疏，λ≈0.87 → capped
        'full':  {'topology': 'full', 'num_conn': 6},     # 稠密，λ=0.5 → off-cap
    }
    probe_arms = {'wc': {'fl_method': 'wc'},
                  'wonly': {'fl_method': 'wc', 'wc_calibrate': 0}}
    # A) 连通度 × 加入点 × 臂（决定性对照）
    for tname, topo in conn.items():
        for frac in (0.5, 0.9):
            for aname, arm in probe_arms.items():
                for seed in SEEDS_SMALL:
                    runs.append((f"{tname}_tau{int(frac*100)}_{aname}_s{seed}",
                                 {**c10, **topo, **s1(T10, frac), **arm, 'seed': seed}))
    # B) c_L 杠杆（稀疏图上，看 c_L=1 能否抬 η_c 脱离 cap）
    for aname, arm in probe_arms.items():
        for seed in SEEDS_SMALL:
            runs.append((f"conn6_cL1_tau50_{aname}_s{seed}",
                         {**c10, **conn['conn6'], **s1(T10, 0.5), **arm, 'c_L': 1.0, 'seed': seed}))
    groups['G8_calib_probe'] = runs

    # ---- G9–G12：off-cap 机制补全（组件 C 的定量验证必须在 √ 分支运行）----
    # G8 结论：稀疏图（conn6，λ≈0.87）上校准恒 cap-active，√ 分支从未执行，
    # 故 conn6 上的 G2/G3/G4/G5-κ 只记录了封顶分支行为。V2、完整 V4 阶梯、
    # E8 κ 鲁棒性、E9 调度对比须在 full 拓扑（λ=0.5，off-cap）重做。
    full_base = {**c10, 'topology': 'full', **s1(T10, 0.5), 'fl_method': 'wc'}

    # G9 = V2 off-cap：η 网格 {η_c·2^-j} + 校准参照（判据：M(η̂)/M(η_best) ≤ (c+1/c)/2×1.25）
    runs = []
    for j in range(7):
        for seed in SEEDS_SMALL:
            runs.append((f"gridfull_j{j}_s{seed}",
                         {**full_base, 'wc_eta_frac': 0.5 ** j, 'seed': seed}))
    for seed in SEEDS_SMALL:
        runs.append((f"gridfull_calibrated_s{seed}", {**full_base, 'seed': seed}))
    groups['G9_grid_full'] = runs

    # G10 = V4 阶梯 off-cap：τ_k 全扫描（G8 仅 2 点已验证方向，此处出论文阶梯图）
    runs = []
    for frac in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for seed in SEEDS_SMALL:
            runs.append((f"taufull{int(frac * 100)}_s{seed}",
                         {**c10, 'topology': 'full', **s1(T10, frac),
                          'fl_method': 'wc', 'seed': seed}))
    groups['G10_tau_full'] = runs

    # G11 = E8/R5 off-cap：κ 扰动（capped 下 κ 被 min 吞掉，G5 的 κ 臂无效；
    # 判据：M 膨胀 ≤ (√κ+1/√κ)/2；κ=1 参照 = G9 gridfull_calibrated）
    runs = []
    for kappa in (0.0625, 0.25, 4.0, 16.0):
        for seed in SEEDS_SMALL:
            runs.append((f"kappafull{kappa}_s{seed}",
                         {**full_base, 'wc_kappa_g': kappa, 'seed': seed}))
    groups['G11_kappa_full'] = runs

    # G12 = E9 off-cap：调度形状 vs 常数 η̂（常数参照 = G9 gridfull_calibrated）
    runs = []
    for sched in ('sqrt_decay', 'cosine', 'warmup'):
        for seed in SEEDS_SMALL:
            runs.append((f"schedfull_{sched}_s{seed}",
                         {**full_base, 'wc_post_schedule': sched, 'seed': seed}))
    groups['G12_sched_full'] = runs

    # ---- G13/G14：主表 v2（cifar100/tiny，增强 + 调好的地板 lr）----
    # 差距审计结论：固定 lr 臂必须 per-dataset 调参（"打败未调参 baseline"不成立），
    # 且绝对精度须进入文献可比区间（补数据增强，理论中性）。cifar10 与全部机制组
    # （G2–G12）不受影响、不重跑。老的 G1_main_cifar100/tiny（无增强、lr=0.01、
    # ResNet 从头训练）保留在盘上作 no-aug 记录，论文不再使用。
    for ds_key in ('cifar100', 'tiny_imagenet'):
        ds = DATASETS[ds_key]
        runs = []
        for lr in GRID_LRS:
            for seed in SEEDS_SMALL:
                runs.append((f"lr{lr}_s{seed}",
                             {**COMMON, **ds, **s1(ds['n_rounds']), **ARMS['w_only'],
                              'lr': lr, 'augment': 1, 'seed': seed}))
        groups[f'G13_lrgrid_{ds_key}'] = runs

    # G14：lr* 自动解析自 G13 落盘结果（或 LR_STAR 手动覆盖）；网格未完成时不生成。
    # 全部臂共享 lr*（含 wc 的 Phase-1）——"调好的地板上 wc 额外自校准"。
    for ds_key, scen_names, seeds in (('cifar100', ['S1', 'S2'], SEEDS),
                                      ('tiny_imagenet', ['S1'], SEEDS)):
        lr_star = resolve_lr_star(ds_key)
        if lr_star is None:
            continue
        ds = DATASETS[ds_key]
        runs = []
        all_settings = {'S1': s1(ds['n_rounds']), 'S2': s2(ds['n_rounds'])}
        for scen in scen_names:
            for arm_name, arm in ARMS.items():
                for seed in seeds:
                    runs.append((f"{ds_key}_{arm_name}_{scen}_s{seed}",
                                 {**COMMON, **ds, **all_settings[scen], **arm,
                                  'lr': lr_star, 'augment': 1, 'seed': seed}))
        groups[f'G14_main_aug_{ds_key}'] = runs

    # ---- G15：CIFAR-10 的地板 lr 选择（消除主表列间的调参不对称）----
    # CIFAR-10 的 η_pre=0.01 是 LeNet 默认值、从未调过，而 cifar100/tiny 经 G13 调过。
    # 用 CIFAR-10 自己的（无增强）设定跑同样的四常数选择，去掉这个不对称。
    # 注意与 G2 的区别：G2 扫 η_c·2^-j（顶到 0.0019，研究 M(η) 在容许上限附近的形状），
    # 本组扫绝对常数（含 0.01/0.03，研究 η_pre 的选择），两者覆盖不相交的区间。
    runs = []
    for lr in GRID_LRS:
        for seed in SEEDS_SMALL:
            runs.append((f"lr{lr}_s{seed}",
                         {**COMMON, **DATASETS['cifar10'], **s1(T10),
                          **ARMS['w_only'], 'lr': lr, 'seed': seed}))
    groups['G15_lrgrid_cifar10'] = runs

    # ---- G16：用去中心化谱隙估计替代精确特征分解 ----
    # 论文诚实声明：全部数字的 λ̂ 来自全谱特征分解，而部署中无客户端持有 W^t；
    # 稀疏图上 η̂=(1−λ̂)/(7L̂) 恒等于封顶值，故整条规则在该分支上是 λ̂ 的函数。
    # 幂迭代（m 轮 gossip）低估 λ → 抬高 η_c（越界风险）；surrogate 高估 → 安全但保守。
    # exact 行由既有 G2/G8 的 conn6 默认配置提供，此处只跑替代估计。
    runs = []
    for est in ('power5', 'power10', 'power20', 'surrogate'):
        for seed in SEEDS_SMALL:
            runs.append((f"specest_{est}_s{seed}",
                         {**COMMON, **DATASETS['cifar10'], **s1(T10),
                          'fl_method': 'wc', 'lambda_estimator': est, 'seed': seed}))
    groups['G16_specest'] = runs

    # ================= 内审补充实验 E1–E4 =================
    # 共同点：全部使用固定步长臂或成对的校准臂，指标以 across-seed 标准差为判据
    # （五个配对样本下 Wilcoxon 最小可达 p=0.0625，故不报 p 值）。

    # ---- E1：无迟到者对照（论文标题级主张目前零实验支撑）----
    # 摘要/引言/§3.2 都称 floors "正是无迟到者网络已付的"，但没有任何一次 run 是
    # 全体客户端从第 0 轮在场的。四个配置共享划分、图与客户端 k（RNG 已对齐），
    # 只有 join 时刻不同。**必须用固定步长臂**：A0 无 join → 永不校准，若用 wc 臂
    # 则 A0 停在 0.01 而 A1/A2 切到 ~0.0019，5 倍步长差会直接污染 floor 比较
    # （floor ∝ η）。故全部用 w_only（warm init + 固定步长），A0/A3 下退化为 D-PSGD。
    # 比较落在尾窗 M_tail=[0.9T,T)：A0 从 0 轮就用全部数据，全窗必然领先，
    # 那是 term (i) inherited gap，不是 floor。
    runs = []
    e1_base = {**COMMON, **DATASETS['cifar10'], **ARMS['w_only']}
    for cfg, extra in (
            ('A0_nojoin', {'temp_client_dist': 'none', 'minimum_join_rounds': 0}),
            ('A1_join30', {**s1(T10, 0.3)}),
            ('A2_join60', {**s1(T10, 0.6)}),
            # A3：k 永不加入——网络下降 f_{n−1} 而度量 f_n（目标改变的可视化），
            # 故必须打开 metric_all_clients，否则测的是 f_{99}，与其余三配置不可比
            ('A3_never', {**s1(T10, 1.0), 'metric_all_clients': 1}),
    ):
        for seed in SEEDS:
            runs.append((f"{cfg}_s{seed}", {**e1_base, **extra, 'seed': seed}))
    groups['E1_nojoiner'] = runs

    # ---- E2：把 M-tuned floor 的网格补完 ----
    # A9 自己写着 2.34 / 1.54 是 M-tuned floor 的**上界**而非 floor 本身，
    # "how far it fails to survive we do not know"——开着这个洞交上去，审稿人会直接
    # 读成"规则在它自己针对的指标上没有收益"。向上延伸定步长网格定位内部极小。
    # 停止规则（事前固定）：连续两点 M 上升 / 准确率跌破 no-collab 参照
    # （CIFAR-100 0.382，Tiny 0.306）/ 出现 NaN 即发散——三者皆为可接受的终点。
    runs = []
    for ds_key in ('cifar100', 'tiny_imagenet'):
        ds = DATASETS[ds_key]
        for lr in (0.1, 0.3):
            for seed in SEEDS_SMALL:
                runs.append((f"{ds_key}_lr{lr}_s{seed}",
                             {**COMMON, **ds, **s1(ds['n_rounds']), **ARMS['w_only'],
                              'lr': lr, 'augment': 1, 'seed': seed}))
    groups['E2_mgrid'] = runs

    # ---- E3：n-sweep（把 init rule 从 null result 变成 positive result）----
    # 目的不是验 (n−1)/n² 的算术（恒等式已验到 1e-15），而是三件事：
    #   (1) 效应随 n 减小而**出现**，比"到处看不见"强得多；
    #   (2) 测 prop:calib 的 n 消去——论文亲口写着该主张 "rests on no measurement of ours"；
    #   (3) 区分 n^{-1} 与 n^{-2}（两条律只在 n=100 交会，正是唯一测过的点）。
    # 协议改动（须在论文声明）：固定 per-client shard=250（train 200 + val 50），
    # 每个 n 单独抽一次 Dirichlet——现协议固定总量 50k，n=20 每轮 39 个 minibatch 而
    # n=200 只有 4 个，十倍的 per-round 工作量差会淹没 n 的效应。固定 shard 后
    # batch=64 + drop_last 使每个 n 都恰好 3 个 minibatch，工作量自动相等，
    # 故**不需要**额外的 local_steps 参数。n=200 恰好用满 CIFAR-10 的 50000。
    # 不跑任何外部 baseline（DFedAvg/EL/no-collab 都不定义 join），但 cold 对照必跑：
    # init rule 的全部主张都是配对差，孤立的 warm Ω^{τ_k} 不可解释。
    runs = []
    e3_configs = {
        'C1_cold_fixed': ARMS['cold'],                                  # shock 与可见性的配对基准
        'C2_warm_fixed': ARMS['w_only'],
        'C3_warm_calib': ARMS['wc'],                                    # 真实回退路径：η̂ ∝ √n
        'C4_warm_calib_forced': {**ARMS['wc'], 'wc_force_eps1_zero': 1},  # 强制 plateau 分支：n 应消去
    }
    for n_clients in (20, 50, 100, 200):
        for cfg_name, cfg in e3_configs.items():
            for seed in SEEDS_SMALL:
                runs.append((f"n{n_clients}_{cfg_name}_s{seed}",
                             {**COMMON, **DATASETS['cifar10'], **s1(T10, 0.5), **cfg,
                              'split_method': f'dirfix{n_clients}', 'seed': seed}))
    groups['E3_nsweep'] = runs

    # ---- E4：staggered 反转的诊断（2×2，其中一维是零成本重分析）----
    # app:extra:s2 报告最现实场景里 ΔM 从 −0.70 翻成 +0.67，并明说两个候选解释
    # "not separated by any measurement we have"。两维：
    #   到达窗口：现状 (0.25T,T]（首个到达落在 precondition 失效区）→ 新 (0.5T,0.8T]
    #             （全部落在 plateau 区，且尾部留出成员固定窗）——需要新 run；
    #   M 的窗口：从首个到达起算（现状）→ 从最后一个到达起算（成员固定）——
    #             summary 已落盘 M_window_last_join，**零成本重分析**。
    # 四种结局都可写，这是本轮风险最低的一组。
    runs = []
    ds = DATASETS['cifar100']
    lr_star = resolve_lr_star('cifar100')
    for arm_name in ('wc', 'w_only', 'c_only', 'cold'):
        for seed in SEEDS:
            runs.append((f"lateS2_{arm_name}_s{seed}",
                         {**COMMON, **ds, **ARMS[arm_name], 'augment': 1,
                          'lr': lr_star if lr_star else 0.01,
                          'temp_client_dist': 'uniform', 'delay_client_ratio': 0.2,
                          'minimum_join_rounds': int(0.5 * ds['n_rounds']),
                          'join_round_max': int(0.8 * ds['n_rounds']),
                          'seed': seed}))
    groups['E4_stagger_late'] = runs

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
        for ds_key in ('cifar100', 'tiny_imagenet'):
            if f'G14_main_aug_{ds_key}' not in all_groups:
                print(f"  G14_main_aug_{ds_key:14s} （待 G13_lrgrid_{ds_key} 完成后自动生成）")
        return

    failed = []
    for group in args.groups:
        # 每组开跑前重建组列表：G14 依赖 G13 的落盘结果自动解析 lr*，
        # 使 `run_experiments.py G13_... G14_...` 一条命令可以完整跑通流水线
        all_groups = build_groups()
        if group not in all_groups:
            if group.startswith('G14'):
                print(f"跳过 {group}：对应的 G13 网格尚未全部完成，无法自动解析 lr*"
                      f"（先跑 G13，或在 LR_STAR 手动填值）")
            else:
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
