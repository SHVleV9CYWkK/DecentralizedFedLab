#!/usr/bin/env bash
# 内审补充实验 E1–E4（EXPERIMENT_PLAN_v3）。幂等：已完成的 run 自动跳过，可随时中断重跑。
#
#   bash run_supplementary.sh            # 按优先级依次跑完 E1→E2→E3→E4
#   bash run_supplementary.sh E1 E2      # 只跑指定组（"只能跑一组就跑 E1"）
#   DFL_DEVICE=cuda bash run_supplementary.sh
#
# 组 → 规模 → 修复什么
#   E1  20 runs  无迟到者对照：标题级主张（floors 即无迟到者网络已付的）目前零实验支撑
#   E2  12 runs  M-tuned floor 网格：A9 自己标出"我们不知道差多远"的洞
#   E3  48 runs  n-sweep：把 init rule 从 null result 变成 positive；测 prop:calib 的 n 消去
#   E4  20 runs  staggered 反转诊断：分离两个候选解释（另一维是零成本重分析）
set -u
cd "$(dirname "$0")"

declare -A GROUP=(
  [E1]="E1_nojoiner" [E2]="E2_mgrid" [E3]="E3_nsweep" [E4]="E4_stagger_late"
)
ORDER=(E1 E2 E3 E4)
[ $# -gt 0 ] && ORDER=("$@")

# E3 需要固定 shard 的切分（每客户端 200 train + 50 val，n=200 恰好用满 CIFAR-10）。
# 固定 shard 后每个 n 都恰好 3 个 minibatch/轮，per-round 工作量自动相等，
# 因此只有 n 在变——这也是不需要额外 local_steps 参数的原因。
if printf '%s\n' "${ORDER[@]}" | grep -qx E3; then
  for n in 20 50 100 200; do
    if [ ! -d "client_indices/cifar10_dirfix${n}_0.4" ]; then
      echo "[prep] 生成 n=${n} 的固定 shard 切分"
      python build_statistical_heterogeneity.py --dataset_name cifar10 --clients_num "$n" \
        --split_method "dirfix${n}" --alpha 0.4 --shard_size 250 --test_ratio 0.2 \
        --seed 42 --dataset_indexes_dir client_indices >/dev/null || exit 1
    fi
  done
fi

for tag in "${ORDER[@]}"; do
  g="${GROUP[$tag]:-}"
  if [ -z "$g" ]; then echo "未知组: $tag（可选 ${!GROUP[*]}）"; continue; fi
  echo "==================== $tag / $g ===================="
  python run_experiments.py "$g" || echo "[warn] $tag 有失败的 run，继续下一组"
done

echo "==================== 聚合 ===================="
python -m analysis.aggregate
cat <<'EOF'

分析要点（判据在看到数据前已固定，全部以 across-seed 标准差陈述，不报 p 值）：

  E1  P10a  M_tail(A0/A1/A2) 两两之差 ≤ 三者合并的 across-seed s.d.
      P10b  g(w)=M_w(A1)−M_w(A0) 随窗口后移单调下降，末窗落进 s.d.
      注意  比较只能落在尾窗。A0 从 0 轮就用全部数据，全窗必然领先——那是
            term (i) inherited gap，不是 floor。A3 用 metric_all_clients 测同一个 f_n。
      写法  floor 是上界而非对实测值的预言，P10a 成立只能说"该主张的经验内容成立"。

  E2  P13   定步长网格上 M 有内部极小。若准确率崩溃/发散前 M 始终单调下降则 refuted。
      两种结局都能写：找到 M-optimal → 换比较基准；M-optimal 明显更优 → 改报
      c=η̂/η_best 与包络 (c+1/c)/2，这是规则本来就只能支撑的主张（CIFAR-10 已在这么写）。

  E3  P11   log E_n 对 log n 回归斜率 ∈ [−1.3,−0.7]，且 R_n=D_k²/Ω^{τ_k−} 极差 ≤ 2×。
            若 R_n 极差 > 2× → sweep 被混杂，改报 R_n 本身而不报斜率（事前写死，防 p-hacking）。
      P12   |ΔM|/s.d. 随 n 减小单调上升；n=20 时 >1，n=200 时 ≪1。必须报归一化比值——
            n=20 的 variance floor 大 5 倍，原始 ΔM 不可比。
      P14   C4（gate_path=forced）的 η̂ 跨 n 变化 ≤ s.d.；C3（fallback）的 η̂ 随 n 上升。
            两条曲线一起报——主动说出"回退路径不是 n-free"，而不是等审稿人算出来。
      披露  逐 n 报 λ̂（degree/n 从 0.3 变到 0.03）；总数据量随 n 增长，跨 n 的 M 绝对值
            标为 observational。

  E4  2×2：到达窗口（本组新 run，(0.5T,0.8T]）× M 的窗口（零成本：summary 里的
      M_window vs M_window_last_join）。四种结局都可发表，风险最低。

⚠ 跑 E3 之前先改摘要：join shock 是 (n−1)/n²·D_k² = Θ(n⁻¹)，不是 Θ(n⁻²)
   （Θ(n⁻²) 只属于平均位移的平方，而该量不进 eq:phase2 任何一项）。
   E3 会产出 n⁻¹ 的数据，摘要不改的话审稿人会把你的实验读成对你自己摘要的反驳。
EOF
