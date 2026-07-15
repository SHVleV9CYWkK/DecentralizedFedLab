# 实验启动指南：脚本用法与内部机制

本文档说明如何启动实验（单个 run 与批量矩阵），以及启动链路上每个脚本/模块
各自做了什么。结果的统计与分析见姊妹文档 [results_analysis.md](results_analysis.md)。

---

## 1. 环境准备

```bash
conda activate FL          # 含 torch / timm / pandas / scipy / pyarrow
cd <项目根目录>
```

前置数据（仓库已具备，列出仅供核对）：

- `data/`：CIFAR-10 / CIFAR-100 / EMNIST / Tiny-ImageNet 原始数据；
- `client_indices/<数据集>_dirichlet_<α>/client_<i>/{train,val}_indexes.npy`：
  非 IID 切分索引（cifar10、emnist 为 100 客户端；cifar100、tiny_imagenet 为 50）。
  新的 α 或客户端数用 `build_statistical_heterogeneity.py` 生成。

---

## 2. 快速开始

### 2.1 批量跑实验矩阵（推荐入口）

```bash
# 1) 设定设备：编辑 run_experiments.py 顶部的全局变量
#    DEVICE = "cuda"   # Apple 笔记本: "mps"；服务器: "cuda"；调试: "cpu"
#    或者不改文件，用环境变量临时覆盖：
DFL_DEVICE=cuda # python run_experiments.py G1_main_cifar10

# 2) 查看全部实验组与完成进度
python run_experiments.py --list

# 3) 检查将要执行的命令（不实际运行）
python run_experiments.py G2_grid --dry_run

# 4) 跑完后聚合
python -m analysis.aggregate
```

### 2.2 单独跑一个自定义 run

```bash
python main.py \
  --fl_method wc --dataset_name cifar10 --model lenet \
  --alpha 0.4 --split_method dirichlet \
  --n_rounds 200 --minimum_join_rounds 100 --temp_client_dist single \
  --optimizer_name sgd --scheduler_name constant --lr 0.01 \
  --symmetry 1 --gossip 0 --num_conn 6 --n_job 1 --eval_every 5 \
  --seed 42 --device cuda \
  --exp_group my_test --run_name wc_tau100_s42
```

`--exp_group/--run_name` 决定日志目录 `results/<exp_group>/<run_name>/`；
`run_name` 留空则用时间戳。已 COMPLETED 的 (exp_group, run_name) 会被直接跳过。

---

## 3. `run_experiments.py`：实验矩阵启动器做了什么

### 3.1 职责

把实验清单（E1–E20）翻译成 **311 个具体的 `main.py` 调用**，并提供：

- **分组**：每个实验组对应清单中的一类问题（见 §3.4 的对应表）；
- **幂等续跑**：启动每个 run 前检查 `results/<组>/<run>/summary.json` 是否
  `COMPLETED`，是则跳过——崩溃/断电/手动中断后重新执行同一命令即可续跑；
- **失败隔离**：单个 run 失败不中断矩阵，结束时汇总打印失败清单；
- **设备解耦**：训练设备由脚本顶部 `DEVICE` 全局变量（或 `DFL_DEVICE` 环境
  变量）统一控制，实验配置与机器无关，同一份脚本在 Mac（mps）与服务器
  （cuda）上行为一致。

### 3.2 配置结构（改实验时只动这四处）

| 常量 | 内容 |
|---|---|
| `MAIN_SEEDS` | 主表**按数据集**分配种子：**cifar10/cifar100 用 5 个**（统计骨架，完整 Wilcoxon 显著性），**emnist/tiny 用 3 个**（泛化广度）。cifar100 墙钟比 emnist 快（50 客户端 + GPU 高效 ResNet）故承担骨架。`SEEDS_SMALL`（3 个）供机制/敏感性实验。**同一种子下所有臂的延迟时间表完全相同**（配对比较）；加种子 = 在列表**末尾追加**后重跑（幂等，只补新种子、不重排） |
| `MAIN_SCENARIOS` | 主表**按数据集**分配加入场景：**cifar10/cifar100 跑 `S1+S2`**（骨架），**emnist/tiny 只跑 `S1`**（广度）。消融 4 臂在所有数据集上完整保留 |
| `COMMON` | 全部 run 共享的地板参数（GR1 同地板比较）：纯 SGD、常数调度、对称固定图、`lr=0.01`、`n_job=1` 等 |
| `DATASETS` | 数据集 × 模型 × **轮数**：cifar10/LeNet/200、cifar100/ResNet18-GN/200、emnist/LeafCNN1/150、tiny/TinyViT/200。**轮数须足够长，使 Phase-1 在 τ_k=0.5T 前进入平台（Δ̂_k 稳定）、加入后窗口仍脱离 cap-active——这是组件 C 的前提**（短轮数下 V2/V4 不成立，见 §8） |
| `ARMS` | 实验臂 = `fl_method` + WC 消融开关组合：`wc / w_only / c_only / cold / dfedavg / ellocal`。论文标注 cold≡D-PSGD、w_only≡D-PSGD+aggregate-on-join；**dfedsam 已移除**（SAM 依赖动量，与无动量同地板不兼容，三数据集退化至随机） |
| `LR_STAR` | 主表 v2 的调好地板 lr（G13 网格结果填入；None 时 G14 组不生成） |

加入场景由两个辅助函数生成：

- `s1(n_rounds, frac=0.5)`：**S1 受控单事件**——1 个客户端恰在 `frac·T` 轮加入，
  客户端按种子随机抽取；**绑定全部理论（R1 恒等式、单一 τ_k、校准公式）**；
- `s2(n_rounds)`：**S2 错峰多事件**——20% 客户端在 `(0.25T, T]` 均匀抽取的轮次
  顺序加入；现实性场景。

### 3.3 run 命名与目录

`run_name = <数据集>_<臂>_<场景>_s<种子>`（如 `cifar10_wc_S1_s42`），
日志在 `results/<组名>/<run_name>/`。命名是确定性的——这就是幂等跳过的键。

### 3.4 实验组 ↔ 实验清单对应

| 组 | run 数 | 覆盖的实验 | 说明 |
|---|---|---|---|
| `G1_main_cifar10` | 70 | E11 主表、E16 种子、E17/E18 跨数据集模型；E1/E2/E4/E10/E13 纯分析 | 7 臂 × S1+S2 × 5 种子（统计骨架） |
| `G1_main_cifar100` | 70 | 同上（ResNet18-GN） | 7 臂 × S1+S2 × 5 种子（统计骨架） |
| `G1_main_emnist` | 21 | E11/E17/E18 泛化广度（LeafCNN1） | 7 臂 × 仅 S1 × 3 种子 |
| `G1_main_tiny_imagenet` | 21 | E11/E17/E18 泛化广度（TinyViT） | 7 臂 × 仅 S1 × 3 种子 |
| `G2_grid` | 24 | E6 校准 vs 神谕网格（V2） | η̂ := η_c·2⁻ʲ, j=0..6 + 校准参照臂 |
| `G3_tau` | 24 | E7 单调性（V4）、E3 封顶阈值 | τ_k ∈ {0.2..0.9}·T |
| `G4_schedule` | 9 | E9 调度无免费午餐 | sqrt_decay / cosine / warmup |
| `G5_perturb` | 30 | E5 R-W1 违规、E8 κ 扰动、E14 保守方向、E4(c) W-G | fitted / global_sim / κ∈{1/16..16} / c_L / λ̂ 扰动 |
| `G6_topology` | 12 | E15 邻居数、E20 谱隙 | ring / full / num_conn∈{4,10} |
| `G7_alpha` | 30 | E19 异质性 | α ∈ {0.1, 1.0}（α=0.4 复用 G1） |
| `G8_calib_probe` | 30 | 组件 C 诊断（cap-active 根因确认） | wc/w_only × conn6/full/cL1 × τ∈{0.5,0.9}T |
| `G13_lrgrid_*` ×2 | 各 12 | 主表 v2 前置：cifar100/tiny 地板 lr 网格 | w_only × lr∈{0.03..0.001} × 增强 |
| `G14_main_aug_*` ×2 | 60/30 | **主表 v2**：调好 lr* + 增强 + 预训练卷积初始化 | 6 臂；**lr\* 从 G13 落盘结果自动解析**（按平均最终准确率，要求 4 档×全种子完成；`LR_STAR` 手动值可覆盖）。同一条命令 `G13_... G14_...` 可整链跑通 |
| `G9_grid_full` | 24 | **V2 off-cap**（conn6 版被 cap 污染，不可用于 V2） | full 拓扑 η 网格 + 校准参照 |
| `G10_tau_full` | 24 | **V4 阶梯 off-cap**（论文 η̂-τ_k 图） | full 拓扑 τ∈{0.2..0.9}T |
| `G11_kappa_full` | 12 | **E8/R5 off-cap**（capped 下 κ 被 min 吞掉，G5 κ 臂作废） | κ∈{1/16,1/4,4,16} |
| `G12_sched_full` | 9 | **E9 off-cap** 调度 vs 常数 η̂ | sqrt_decay/cosine/warmup |

### 3.5 多机分工

实验组之间无任何共享状态，可按组分配到不同机器；跑完把各机器的
`results/<组名>/` 目录拷到同一处，再 `python -m analysis.aggregate` 即可。
每个 run 的 `config.json` 记录 `git_commit`，可校验各机器代码版本一致。

---

## 4. `main.py`：单个 run 内部发生了什么

每次 `main.py` 调用 = 一个 run，流程：

```
1. 解析参数 → 设随机种子（torch/random/numpy）→ 选设备（cuda 不可用自动回落 cpu）
2. 幂等检查：results/<组>/<run>/summary.json 已 COMPLETED → 直接退出
3. 创建 RunLogger（utils/run_logger.py）并挂接事件总线（utils/event_bus.py）
   ——此后客户端/协调器 emit 的协议事件自动落盘 events.jsonl
4. 加载数据集与模型（utils/utils.py）→ 工厂创建全部客户端（clients/client_factory.py）
5. 抽取延迟时间表 get_client_delay_info（由种子决定 → 同种子各臂相同）
   并写入 config.json 的 client_delay 字段
6. 构造 Coordinator：生成通信图（含连通性核查）、计算精确 λ̂、下发拓扑信息
7. 主循环 r = 0..n_rounds-1：
     train_client(r)        本轮加入的客户端入场（热启动→校准）+ 全员本地训练
     interchange_model(r)   发送快照 → 按图分发 → pre_add 拉取 → 各自混合
     evaluate_client()      每客户端验证集评测 → overall/client 指标
     记录 network 指标      Ω 每轮；‖∇f_n(θ̄)‖² 每 eval_every 轮及加入轮邻域
     lr_scheduler()         （wc 客户端为 no-op：常数步长）
8. 写 summary.json：最终指标、各加入事件的窗口平稳性 M、墙钟、状态
   （异常中断也会写，status=INTERRUPTED，重跑时旧数据被清空重写）
```

### 4.1 Coordinator（coordinator.py）的关键机制

| 机制 | 说明 |
|---|---|
| **连通性核查** | 生成图后验证「Phase-1 在位集合 + 每个加入事件后的累积集合」的诱导子图全部连通（A5 前提；延迟客户端可能是割点）。随机图重试至 200 次，ring/full 不满足直接报错 |
| **精确 λ̂** | 每次成员变更对活跃诱导子图的 lazy Metropolis 矩阵做特征值分解，λ̂ 下发给客户端（研究模式 [T]），同时发 `topology` 事件 |
| **度数下发** | `set_topology_info` 把每个客户端在活跃图上的真实度数告知它（等价规范的 DEGREE 交换），保证 Metropolis 权重两端一致 |
| **发送快照** | 每轮每发送方的模型克隆一次、所有接收方共享冻结副本——保证「先全部交换、后混合」的同步轮语义（式 (U) 次序），消除聚合顺序依赖 |
| **pre_add 拉取** | 加入前一轮（τ_k−1）把邻居模型推给即将加入的客户端 = 规范的 PULL#2 新鲜快照；`--wc_warm_mode global_sim` 时改为推送全部活跃客户端模型（W-G 变体） |
| **加入恒等式采集** | 加入瞬间记录 Ω^{τ_k−}、Ω^{τ_k}、D_k（float64）→ `join_identity` 事件，供 E1/V1 核对 |
| **评测指标** | `consensus_error()`（Ω）与 `stationarity_gradnorm2()`（全批 ‖∇f_n(θ̄)‖²）均按 Algorithm 9 口径 float64 聚合 |

### 4.2 WC 客户端（clients/dfl_method_clients/wc_client.py）每轮做什么

```
train() 入口：
  ① 查 JOIN 公告板：有 τ_k ≤ 当前轮的未生效事件 → 确定性校准 η̂ 并切换（全网逐比特一致）
  ② （E9 调度臂才生效）按 post_schedule 形状更新 lr
  ③ 本地 SGD（强制 momentum=0；local_epochs × 全部批次）
  ④ 平台检测 EMA 更新；每 K_const 轮刷新 L̂（同批次方向探针×c_L）与 σ̂²（独立小批量样本方差）

加入时（set_init_model 收到 pre_add 模型）：
  θ_warm = 收到模型的平均（R-W1：本地拟合权重必须丢弃）
  Δ̂_k = 固定评测集上 loss(θ_warm) − 本地 Adam 拟合可达损失（Algorithm 7，副本上拟合）
  发布 JOIN{τ_k, Δ̂_k, ε̂₁, L̂, σ̂², λ̂_post} 到公告板（模拟洪泛，要求 n_job=1）
```

### 4.3 消融开关（G2–G6 用，单跑也可手动传）

| 参数 | 默认 | 作用（对应实验） |
|---|---|---|
| `--wc_warm_mode` | neighbor | `global_sim`=W-G；`cold`=冷加入；`fitted`=故意违反 R-W1（E5） |
| `--wc_calibrate` | 1 | 0 = 不校准（"仅 W"臂） |
| `--wc_post_schedule` | constant | sqrt_decay / cosine / warmup（E9） |
| `--wc_eta_frac` | 0 | >0 ⇒ η̂ := frac·η_c（E6 网格） |
| `--wc_kappa_g` | 1.0 | Ĝ_n 乘性扰动（E8） |
| `--c_L` | 2.0 | L̂ 保守系数（E14） |
| `--lambda_hat_override` | -1 | >0 ⇒ 强制 λ̂，绕过精确特征值（E14） |

---

## 5. 延迟客户端的设置

延迟时间表**在每个 run 启动时按种子抽取**（`utils/utils.py:get_client_delay_info`），
不需要提前定义；实际抽取结果记录在该 run `config.json` 的 `client_delay` 字段。

| 参数 | 作用 |
|---|---|
| `--temp_client_dist` | `single`：1 个客户端恰在 `minimum_join_rounds` 轮加入；`uniform`：按比例抽一组，加入轮在 `(minimum, T]` 均匀抽取；另有 `even` / `normal` |
| `--minimum_join_rounds` | 最早加入轮（single 模式即精确加入轮） |
| `--delay_client_ratio` | 延迟客户端比例（多客户端模式） |
| `--set_single_delay_client` | ≥0 指定固定客户端；-1 按种子随机 |

性质：**同种子 ⇒ 相同时间表**（抽取发生在种子设定之后、无其他随机消耗之前），
因此同种子的不同臂构成配对比较；不同种子抽到不同延迟客户端，把"谁延迟"
折进统计变异（E10/E16 的设计）。

---

## 6. 重要约束与注意事项

1. **`--n_job 1`**：WC 的 JOIN 公告板靠单进程共享内存模拟洪泛，多进程下失效。
   启动器已固定为 1。
2. **`--symmetry 1 --gossip 0`**：理论假设 A5 要求对称混合矩阵；固定图才有
   精确 λ̂。每轮随机图（`gossip 1`）仅作为时变拓扑消融臂。
3. **全部臂统一无动量 SGD**：WCClient 内部强制 momentum=0（式 (U) 语义）；
   `get_optimizer` 的 sgd 分支也已改为 momentum=0（GR1 同地板——动量使有效步长
   ≈η/(1−β)，0.9 时≈10η，属不同环境）。**momentum 改动之前跑的 baseline 结果
   （dfedavg/dfedsam/ellocal，m=0.9）与新口径不可比，须删除对应 run 目录重跑**：
   `rm -rf results/G1_main_*/*_dfedavg_* results/G1_main_*/*_dfedsam_* results/G1_main_*/*_ellocal_* results/G7_alpha/*_dfedavg_*`
   （幂等启动器只补跑被删的臂，wc 族不受影响、无需重跑）。
4. **BatchNorm 模型不要直接用**：模型平均破坏 BN running stats（规范 §8-2）。
   矩阵中已使用 GN 版本：`resnet18gn`、`tinyvit`（内部 BN→GN）。原始
   `resnet18` 仅保留兼容，不要用于本课题实验。
5. **`eval_every` 的代价**：平稳性梯度范数 = 全网每客户端一次全批前向+反向，
   约等于一轮训练的开销 / eval_every。默认 5；只看准确率的探索性 run 可设
   `--eval_every 0` 关闭（但该 run 将没有 M_window）。
6. **轮数预算**：`DATASETS` 中的 `n_rounds` 须满足组件 C 的平台前提（见 §8）；
   修改后 τ_k、S2 的抽取范围会按比例自动调整（`s1/s2` 按 `n_rounds` 计算）。
7. **改 `n_rounds` 不会触发重跑（footgun）**：run_name 不含轮数，幂等启动器看到旧
   `summary.json` 会跳过，于是拿到旧轮数的陈旧结果。改轮数后必须**先删除/归档
   服务器上对应组的 `results/` 目录**再重跑（`mv results results_old` 或 `rm -rf`）。
8. **新增实验组**：在 `build_groups()` 里仿照现有组追加；run 命名保持确定性
   （含全部变化维度 + 种子），即可获得幂等续跑。

---

## 7. 典型工作流

```bash
# 0) 若是改了轮数后的重跑：先归档旧结果（见 §6-7 footgun）
mv results results_old_invalid     # 或 rm -rf results

# 1) 先只跑一个快组当"平台校验"（cifar100 墙钟快），确认进入平台再铺全量
DFL_DEVICE=cuda python run_experiments.py G1_main_cifar100
python -m analysis.aggregate
#   检查 §8 的平台判据：Δ̂_k 已从 ~2.0 降下来、η̂ 脱离撞顶、cap_active 大幅减少。
#   不达标就加长 n_rounds（或调 local_epochs）再来——别急着铺全量。

# 2) 平台达标后，铺其余主表与机制组（按机器分配）
DFL_DEVICE=cuda python run_experiments.py G1_main_cifar10 G1_main_emnist
DFL_DEVICE=cuda python run_experiments.py G2_grid G3_tau G4_schedule G5_perturb G6_topology G7_alpha

# 3) 最终聚合与分析 → 见 docs/results_analysis.md
python -m analysis.aggregate
cat results/report/checks.csv
```

---

## 8. 组件 C 的平台前提（务必满足，否则 V2/V4 不成立）

组件 C 的校准理论（规范 §5.2）建立在 **Phase-1 平台期** 上：网络在 τ_k 前已基本收敛，
则 ε̂₁≈0、失配 Δ̂_k 稳定、`n` 消去，η̂ 公式才会按理论行为（V2 校准近最优、V4 随 τ_k 递增）。

**短轮数会破坏这个前提**（2026-06 首轮 n_rounds=50/100 的实测教训）：

- 网络在 τ_k 时仍在快速下降 → 平台门 P=0、Δ̂_k 高达 ~2.0 且随 τ_k 持续缩小；
- 结果 η̂ 撞顶在 η_c（cap-active），V4 方向**反转**（η̂ 随 τ_k 递减），V2 失败（η̂ 落在 M 最差端）。

**判据（跑完一组后用 events 核对）**：

1. `Δ̂_k`（wc_join 事件）应随 τ_k 增大而**减小并趋稳**，量级远低于欠训时的 ~2.0；
2. `wc_cap_active` 事件应**大幅减少**（η̂ 落进 √ 公式分支，而非撞顶）；
3. G3 的 η̂ 应随 τ_k **单调递增**（V4）；G2 的 η̂ 应落在 M(η) U 形的**最低点附近**（V2）。

**达不到时的两个杠杆**：(a) 继续加长 `n_rounds`；(b) 提高 `local_epochs`（每轮多走本地步，
用更少轮数到平台，且减少按轮计的梯度范数评测开销；T'=剩余步数已正确处理，不破坏校准）。

> 注：平台门 P 是 100 客户端逐一 AND，即便真平台也可能仍为 0——**看 Δ̂_k/ε̂₁ 是否变小
> 比看 P 是否=1 更可靠**（P=0 时走 ε̂₁ 回退，真平台下 ε̂₁≈0、校准照样正确）。
