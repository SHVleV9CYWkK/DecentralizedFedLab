# 实验结果获取与统计分析指南

本文档描述从「实验跑完」到「论文可用的表格、图与结论」的完整流程：原始日志的结构、
聚合命令、每个产出文件的含义、派生指标的精确定义、实验清单（E1–E20）逐项的
核对方法，以及统计检验的标准做法。

理论依据：《WC方法论\_实现规范\_v1.0》（下称规范）；
度量与比较规则遵循原文 GR1–GR4（同窗口、平稳性语义、界层面陈述、瞬态/持久分类）。

---

## 1. 数据流总览

```
run_experiments.py <组名>             # 跑实验（幂等：已完成自动跳过）
        │  每个 run 产出 4 个自包含文件
        ▼
results/<exp_group>/<run_name>/
    config.json      全部参数 + 派生量（实际延迟时间表、λ̂、git commit）
    metrics.jsonl    每轮指标（overall / client / network 三种 scope）
    events.jsonl     协议事件流（JOIN、η̂ 切换、加入恒等式、拓扑刷新）
    summary.json     结束统计（窗口平稳性 M、最终指标、墙钟、状态）
        │
        ▼
python -m analysis.aggregate          # 一条命令聚合全部 run
        │
        ▼
results/report/
    runs.parquet / rounds.parquet / events.parquet    三张 pandas 表（自定义分析入口）
    runs_summary.csv                                   每 run 一行的总表
    table_<组名>.csv / .md                             分组均值±std 主表
    fig_<组名>_accuracy.png / _omega.png               标准曲线图
    checks.csv                                         E1/E2 自动核对结果
```

---

## 2. 跑实验与查看进度

```bash
# 列出全部实验组与完成进度
python run_experiments.py --list

# 跑一个或多个组（设备由脚本顶部 DEVICE 变量或环境变量控制）
DFL_DEVICE=cuda python run_experiments.py G1_main_cifar10 G2_grid

# 只打印命令不执行（检查配置）
python run_experiments.py G3_tau --dry_run
```

要点：

- **幂等续跑**：run 完成的标志是其 `summary.json` 中 `status == "COMPLETED"`。
  中途崩溃/断电后重新执行同一命令，已完成的 run 秒级跳过，未完成的重跑
  （旧的部分日志会被清空重写，不会出现混合数据）。
- **失败不中断**：某个 run 失败时启动器继续跑后面的，结束时汇总打印失败清单。
- **可分机器并行**：不同实验组可以在不同机器上跑，最后把各机器的
  `results/<组名>/` 目录合并到一处再聚合即可（每个 run 自包含，无共享状态）。

---

## 3. 原始日志格式参考

### 3.1 `config.json`

| 字段 | 含义 |
|---|---|
| `args.*` | 该 run 的全部命令行参数（复现实验的完整配置） |
| `client_delay` | **实际抽取**的延迟时间表 `{客户端id: 加入轮}`（同种子各臂相同 → 配对比较的依据） |
| `lambda_hat_initial` | Phase-1 在位图的精确 λ̂（特征值分解） |
| `git_commit` / `status` / `compute_device` | 溯源信息 |

### 3.2 `metrics.jsonl`（每行一个 JSON，按轮追加）

| scope | 频率 | 字段 |
|---|---|---|
| `overall` | 每轮 | `loss, accuracy, precision, recall, f1`（全网客户端平均） |
| `client` | 每轮×每客户端 | 同上 + `client_id` |
| `network` | 每轮 | `Omega`（共识误差，float64）、`eta`（当前步长）、`lambda_hat`、`n_active`；每 `eval_every` 轮及加入轮邻域额外含 `gradnorm2` = ‖∇f_n(θ̄)‖²（全批、float64） |

### 3.3 `events.jsonl`（协议事件流）

| event | 触发 | 关键字段 | 用途 |
|---|---|---|---|
| `topology` | 成员变更 | `lambda_hat, n_active` | λ̂ 随加入的演化 |
| `join_identity` | 每个加入轮 | `omega_pre, omega_post, D_k{id:值}, n_post` | **E1/V1**：R1 恒等式核对 |
| `wc_join` | 新客户端加入 | `tau_k, Delta_k, eps1, P, L, sigma2, lam_post, warm_mode` | Δ̂_k 与钉死元组溯源（E10 等） |
| `wc_eta_switch` | **每个客户端**切换时 | `client_id, tau_k, eta` | **E2**：全网 η̂ 一致性 |
| `wc_cap_active` | 校准触顶 | `tau_k` | **E3/V4**：封顶阈值核对 |

### 3.4 `summary.json`

| 字段 | 含义 |
|---|---|
| `final.*` | 最后一轮的全网平均指标 |
| `join_rounds` | 全部加入轮列表 |
| `M_window_tau{τ}` | 该加入事件后窗口 `[τ, T)` 内 `gradnorm2` 采样的平均 = **窗口平均平稳性 M**（GR2 认证指标，采样近似） |
| `M_window` | 首个加入事件的 M（聚合器分组用） |
| `wall_time_sec` / `status` | 墙钟与状态 |

---

## 4. 聚合：一条命令得到全部结果

```bash
python -m analysis.aggregate                       # 聚合 results/ 全部组 → results/report/
python -m analysis.aggregate --groups G1_main_cifar10 G2_grid   # 只聚合部分组
python -m analysis.aggregate --results_dir results --out paper_assets   # 自定义输出目录
```

聚合器做四件事：

1. **建三张表**并缓存为 parquet（pyarrow 缺失时自动回落 pickle）：
   - `runs.parquet`：每 run 一行 = 全部 args ⊕ summary ⊕ 派生指标 ⊕ `arm` 标签；
   - `rounds.parquet`：轮次级长表（metrics.jsonl 全量，带 `exp_group/run_name/arm` 键）；
   - `events.parquet`：事件级长表。
2. **算派生指标**（定义见 §5）。
3. **出分组主表**：按 `(dataset_name, arm[, temp_client_dist])` 分组的
   mean / std / count → `table_<组名>.csv`（与 `.md`，需 `tabulate`）。
4. **出标准图**：每组的 accuracy-vs-round（按臂取种子均值曲线）与
   Omega-vs-round（对数轴，看加入冲击尖峰）。

> 自动报警：聚合结束时若有 run 的 E1 残差 > 1% 或 E2 η̂ 不一致，终端直接列出
> run 名——这两个属于**实现错误信号**，必须先排查再谈结果。

---

## 5. 派生指标的精确定义

| 指标 | 定义 | 语义 |
|---|---|---|
| `M_window` | $\frac{1}{|S|}\sum_{r\in S}\text{gradnorm2}(r)$，$S$=窗口 $[\tau_k,T)$ 内的采样轮 | **主验证指标**（GR2）：窗口平均平稳性 |
| `acc_dip` | 加入前基线（τ 前最后 3 轮 overall accuracy 均值）−（$[\tau,\tau+10]$ 窗口内最低 accuracy） | 加入冲击的实用度量（可为负 = 无跌落） |
| `recovery_rounds` | 加入后 overall accuracy 首次 ≥ 基线的轮数 − τ；未恢复 = NaN | 恢复速度 |
| `e1_residual` | 相对残差 $\big\|\Omega^{\tau}-\hat\Omega\big\|/\Omega^{\tau}$，$\hat\Omega=\frac{n_{pre}\Omega^{\tau-}+\sum_j D_j^2-\|D_{sum}\|^2/n}{n}$（R1 的 §11 多客户端合成；$m{=}1$ 退化为 $\frac{n-1}{n}\Omega^{\tau-}+\frac{n-1}{n^2}D^2$；旧日志缺 `D_sum` 的多客户端事件跳过），多事件取最大 | R1 是**恒等式**：应 ≈ 浮点精度（实测 ~1e-16） |
| `e2_eta_nunique` | 各 `tau_k` 上全部客户端上报 η̂ 的去重计数的最大值 | 必须 = 1（§3.4 逐比特一致切换） |
| `Delta_k` / `eta_hat` | 首个加入事件的失配估计与切换步长 | E7/E10 的横轴/纵轴 |

---

## 6. 实验清单逐项核对手册

约定：**所有比较在同窗口 $[\tau_k, T]$、同地板参数下**（GR1）；
"更优"只在平稳性界层面陈述（GR3）；准确率结论标注为经验观察（GR2）。

### 6.1 正确性类（先于一切有效性结论）

| 实验 | 数据来源 | 通过判据 |
|---|---|---|
| **E1 加入恒等式** | `checks.csv` 的 `e1_residual` | 所有 run < 1%（实际应为 1e-15 量级；超出 = 混合次序或加入时刻实现错） |
| **E2 η̂ 一致性** | `checks.csv` 的 `e2_eta_nunique` | 所有 run = 1 |
| **E3 封顶阈值** | `events.parquet` 中 `wc_cap_active` 事件 ↔ 用 `wc_join` 字段手算 $T'<\frac{98\hat L\hat G_n}{\hat\sigma^2(1-\hat\lambda)^2}$ | 标志出现位置与阈值一致（数据来自 G3） |

### 6.2 组件 W（数据：G1 各臂 + G5）

**E4 冲击消除 / V1**：对比 `rounds.parquet` 中 `wc / cold / globalsim` 三臂的 Omega 轨迹
（标准图 `fig_*_omega.png` 已画好）。定量判据用 `events.parquet` 的 `join_identity`：

- W-N（`wc` 臂）：$\Omega^{\tau}\le\frac{n-1}{n}\big(1+\frac{n-1}{n|N_k|}\big)\Omega^{\tau-}\times 1.05$，且 τ 后无 Ω 尖峰；
- W-G（`globalsim` 臂）：$\Omega^{\tau}\approx\frac{n-1}{n}\Omega^{\tau-}$（严格下降，命题 W1）；
- 冷加入（`cold` 臂）：出现尖峰，幅度按 R1 用日志里的 `D_k` 核对到 1% 内（恒等式，已由 e1_residual 覆盖）。

**E5 R-W1 违规**（G5 `fitted_*`）：该臂的 `D_k` 应反弹至失配尺度、Ω 出现尖峰、
`M_window` 劣于 `wc` 臂——证明"丢弃本地拟合权重"规则的必要性。

### 6.3 组件 C（数据：G2 / G3 / G4 / G5）

**E6 校准质量 / V2**（G2）：

```python
import pandas as pd
runs = pd.read_parquet('results/report/runs.parquet')
g2 = runs[runs.exp_group == 'G2_grid']
grid = g2[g2.wc_eta_frac > 0].groupby('wc_eta_frac')['M_window'].mean()   # U 形曲线
cal  = g2[g2.wc_eta_frac == 0]                                            # 校准臂
eta_best = grid.idxmin()        # 网格最优（以 frac 表示）
# c = max(η̂/η_best, η_best/η̂)，判据：M(η̂)/M(η_best) ≤ (c+1/c)/2 × 1.25
```

论文图：`M(η)` 的 U 形曲线 + 标出校准臂 η̂ 的落点。

**E7 单调性 / V4**（G3）：`runs.parquet` 中按 `minimum_join_rounds` 排序看 `eta_hat`，
应单调不减直至封顶；`wc_cap_active` 出现位置与 E3 阈值一致。画 η̂-vs-τ_k 阶梯图。

**E8 误标定鲁棒性**（G5 `kappa*`）：各 κ 臂相对 κ=1（`wc` 臂）的 `M_window` 膨胀比应
$\le\frac{\sqrt\kappa+1/\sqrt\kappa}{2}$：κ∈{4,16} 对应 1.25 / 2.125（κ<1 同式）。

**E9 调度无免费午餐**（G4 vs G1 的 `wc` 臂）：sqrt_decay / cosine / warmup 三臂的
`M_window` 都不应显著优于常数 η̂（命题 C1 的实验形式）。

### 6.4 概念与主表

**E10 失配而非距离**（G1 种子维度的免费分析）：不同种子延迟不同客户端 →
`wc_join` 事件给出 Δ̂_k 谱系。对每个 run 计算延迟客户端与全网的分布距离
（如标签分布的 TV/JS 距离，从 `client_indices` 直接算），做
（Δ̂_k, 距离）→ M_window / acc_dip 的偏相关：瞬态应跟随 Δ̂_k 而非距离。

**E11/E16/E17/E18 主表**（G1 × 4 数据集）：`table_G1_main_*.csv` 直接给出
各臂 × S1/S2 的 `M_window / final_accuracy / acc_dip / recovery_rounds` 的
mean±std（count 列 = 种子数，应为 5）。论文叙事：

- 验证轨（定理绑定）：`M_window` 排序 + Ω 轨迹；
- 实用轨（经验观察，明确标注）：`acc_dip`、`recovery_rounds`、最差客户端
  accuracy（从 `rounds.parquet` 的 client scope 取每轮 min 再聚合）。

**E13 平台门**（G1 分析）：`wc_join` 事件含 `P` 与 `eps1`；按 P=0/1 分组对比校准质量。

**E14 估计器保守方向**（G5 `cL*` / `lam*`）：高估方向（c_L=4、λ̂=0.99）只应付出
常数代价；低估方向（c_L=1 偏小、λ̂=0.5 低估）可能破坏 η_c 前提 →
观察 Ω 与 loss 是否震荡/发散（这正是要展示的非对称性）。

**E15/E20 拓扑**（G6）：不同拓扑的 `lambda_hat_initial`（config 直读）↔ η_c ↔
`M_window`；ring（λ̂ 大）应触发更小的 η_c 与更明显的封顶。

**E19 异质性**（G7）：α∈{0.1,1.0} 与 G1 的 α=0.4 合并，看结论方向是否随
异质性保持。

---

## 7. 统计检验（论文显著性声明的标准做法）

关键事实：**同种子的各臂共享同一延迟时间表**（`config.json` 的 `client_delay`
可验证），因此臂间比较是**配对设计**，应使用配对检验（功效远高于独立样本检验）：

```python
import pandas as pd
from scipy import stats

runs = pd.read_parquet('results/report/runs.parquet')
g1 = runs[(runs.exp_group == 'G1_main_cifar10') & (runs.temp_client_dist == 'single')]

pivot = g1.pivot_table(index='seed', columns='arm', values='M_window')
# 配对 t 检验（种子数少时建议同时报 Wilcoxon 符号秩）
t, p = stats.ttest_rel(pivot['wc'], pivot['dfedavg'])
w, pw = stats.wilcoxon(pivot['wc'], pivot['dfedavg'])
print(f'paired t: p={p:.4f}; wilcoxon: p={pw:.4f}')
```

报告规范：

- 每个数字报 **mean ± std（n=5 种子）**，主表已自动给出；
- 多臂两两比较时注意多重比较（Bonferroni 或只报预先声明的关键对比：
  `wc vs cold`、`wc vs w_only`、`wc vs 最强 baseline`）；
- 5 个种子的 Wilcoxon 最小可达 p=0.0625，若需 p<0.05 的符号秩结论，
  把关键对比的种子数加到 8–10（启动器的 `SEEDS` 列表加种子重跑即可，幂等）。

---

## 8. 自定义分析与画图（notebook 入口）

三张 parquet 是全部自定义分析的入口，无需再碰原始 JSONL：

```python
import pandas as pd
runs   = pd.read_parquet('results/report/runs.parquet')    # run 级：配置+summary+派生
rounds = pd.read_parquet('results/report/rounds.parquet')  # 轮次级长表
events = pd.read_parquet('results/report/events.parquet')  # 事件级

# 例 1：某组各臂的 accuracy 轨迹（带种子间标准差带）
g = rounds[(rounds.exp_group=='G1_main_cifar10') & (rounds.scope=='overall')]
curves = g.groupby(['arm','round'])['accuracy'].agg(['mean','std']).reset_index()

# 例 2：加入轮邻域的 Ω 放大图（看冲击尖峰）
net = rounds[(rounds.exp_group=='G1_main_cifar10') & (rounds.scope=='network')]

# 例 3：η̂ vs τ_k（E7）
g3 = runs[runs.exp_group=='G3_tau']
mono = g3.groupby('minimum_join_rounds')['eta_hat'].agg(['mean','std'])

# 例 4：最差客户端准确率（公平性维度）
cl = rounds[(rounds.scope=='client')]
worst = cl.groupby(['exp_group','arm','run_name','round'])['accuracy'].min()
```

---

## 9. 故障排查

| 现象 | 处理 |
|---|---|
| `--list` 显示某组完成数不增长 | 看该组目录下未完成 run 的 `config.json`（`status: INTERRUPTED`）；直接重跑该组，启动器只补跑未完成的 |
| 聚合器报 ⚠ E1/E2 | 实现级错误：先查该 run 的 `events.jsonl`（join_identity / wc_eta_switch），不要使用其结果 |
| `M_window` 缺失 | 该 run `eval_every<=0` 或窗口内无 gradnorm2 采样；确认 `--eval_every 5` |
| 想重跑某个 run | 删除其目录（或其 `summary.json`）再执行启动器 |
| 多机合并后聚合 | 直接合并 `results/` 子目录；run 自包含，`git_commit` 字段可校验代码版本一致性 |
| 报告里数字与图对不上 | 重跑 `python -m analysis.aggregate`（report 是派生物，永远以 results/ 原始数据为准） |

---

## 10. 论文写作的红线提醒（GR1–GR4）

1. **窗口对齐**：一切对比从 τ_k 起算到 T；S2 多事件场景注明使用首事件窗口或分事件报告。
2. **指标语义**：`M_window` 是定理认证的指标；accuracy 系列指标一律标注
   "经验观察"，不得写成"由界推出"。
3. **界层面措辞**："更优"指上界排序的方向性证据，不是真误差排序。
4. **不可触碰项**：方差地板与拓扑地板与加入无关，任何臂都不应声称改善它们——
   若实验中观察到地板差异，那是 η̂ 不同带来的定价差异，须如此解释。
