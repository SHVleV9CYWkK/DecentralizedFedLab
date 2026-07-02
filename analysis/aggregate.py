"""聚合 results/ 下全部 run，产出可直接用于论文的表与图。

用法：
    python -m analysis.aggregate                     # 聚合 results/，输出到 results/report/
    python -m analysis.aggregate --results_dir X --out Y
    python -m analysis.aggregate --groups G1_main_cifar10 G2_grid   # 只聚合部分实验组

产出（results/report/）：
    runs.parquet / rounds.parquet / events.parquet   三张缓存表（pandas 直接读）
    runs_summary.csv                                  每 run 一行：配置 ⊕ summary ⊕ 派生指标
    table_<group>.csv / .md                           按 (dataset, arm) 分组的均值±std 主表
    fig_<group>_accuracy.png / fig_<group>_omega.png  标准曲线图（按臂聚合、种子平均）
    checks.csv                                        验证类核对：E1 恒等式残差、E2 η̂ 一致性

派生指标（自动计算）：
    M_window        加入后窗口平均平稳性（summary 直读，GR2 认证指标）
    acc_dip         加入轮邻域的准确率跌落深度（加入前基线 − 窗口内最低）
    recovery_rounds 恢复到加入前基线所需轮数（未恢复 = NaN）
    e1_residual     R1 恒等式相对残差 |Ω_post − 预测| / Ω_post（应 ≈ 0）
    e2_eta_nunique  τ_k 轮各客户端 η̂ 的去重计数（应 = 1）
"""
import argparse
import glob
import json
import math
import os

import pandas as pd

# 参与 arm 标签的配置键（默认值不显示，保持标签紧凑）
ARM_DEFAULTS = {
    'wc_warm_mode': 'neighbor',
    'wc_calibrate': 1,
    'wc_post_schedule': 'constant',
    'wc_eta_frac': 0.0,
    'wc_kappa_g': 1.0,
    'c_L': 2.0,
    'lambda_hat_override': -1.0,
    'topology': 'random',
}


def arm_label(args):
    label = args.get('fl_method', '?')
    if label != 'wc':
        return label
    parts = []
    for key, default in ARM_DEFAULTS.items():
        value = args.get(key, default)
        if value != default:
            parts.append(f"{key.replace('wc_', '')}={value}")
    return 'wc' if not parts else 'wc[' + ','.join(parts) + ']'


def load_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # 截断的最后一行（run 中断）
    return records


def scan_runs(results_dir, groups=None):
    runs, rounds, events = [], [], []
    for config_path in sorted(glob.glob(os.path.join(results_dir, '*', '*', 'config.json'))):
        run_dir = os.path.dirname(config_path)
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
        exp_group = config.get('exp_group', os.path.basename(os.path.dirname(run_dir)))
        if groups and exp_group not in groups:
            continue
        run_name = config.get('run_name', os.path.basename(run_dir))
        args = config.get('args', {})

        row = {'exp_group': exp_group, 'run_name': run_name,
               'status': config.get('status'), 'git_commit': config.get('git_commit'),
               'arm': arm_label(args), 'run_dir': run_dir}
        row.update(args)
        # config 顶层的派生量（save_config 写入，如 lambda_hat_initial、client_delay）
        skip = {'args', 'status', 'git_commit', 'exp_group', 'run_name',
                'started_at', 'compute_device'}
        for k, v in config.items():
            if k not in skip and k not in row:
                row[k] = json.dumps(v) if isinstance(v, dict) else v  # dict 列 parquet 不稳，存 JSON 串

        summary_path = os.path.join(run_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, encoding='utf-8') as f:
                summary = json.load(f)
            row['status'] = summary.get('status', row['status'])
            for key, value in summary.items():
                if key == 'final' and isinstance(value, dict):
                    for mk, mv in value.items():
                        row[f'final_{mk}'] = mv
                elif key != 'status':
                    row[key] = value
        runs.append(row)

        keys = {'exp_group': exp_group, 'run_name': run_name, 'arm': row['arm']}
        for rec in load_jsonl(os.path.join(run_dir, 'metrics.jsonl')):
            rec.update(keys)
            rounds.append(rec)
        for rec in load_jsonl(os.path.join(run_dir, 'events.jsonl')):
            rec.update(keys)
            events.append(rec)

    return pd.DataFrame(runs), pd.DataFrame(rounds), pd.DataFrame(events)


def derive_run_metrics(runs_df, rounds_df, events_df):
    """按 run 计算 acc 跌落/恢复、E1 恒等式残差、E2 η̂ 一致性。"""
    extra = []
    for _, run in runs_df.iterrows():
        key = (run['exp_group'], run['run_name'])
        rec = {'exp_group': key[0], 'run_name': key[1]}
        rr = rounds_df[(rounds_df['exp_group'] == key[0]) & (rounds_df['run_name'] == key[1])] \
            if len(rounds_df) else pd.DataFrame()
        ev = events_df[(events_df['exp_group'] == key[0]) & (events_df['run_name'] == key[1])] \
            if len(events_df) else pd.DataFrame()

        join_rounds = run.get('join_rounds')
        tau = join_rounds[0] if isinstance(join_rounds, list) and join_rounds else None

        # —— 准确率跌落深度与恢复轮数（实用轨指标）——
        if tau is not None and len(rr):
            overall = rr[rr['scope'] == 'overall'].sort_values('round')
            pre = overall[overall['round'] < tau].tail(3)['accuracy']
            post = overall[overall['round'] >= tau]
            if len(pre) and len(post):
                baseline = pre.mean()
                window = post[post['round'] <= tau + 10]
                rec['acc_dip'] = baseline - window['accuracy'].min() if len(window) else None
                recovered = post[post['accuracy'] >= baseline]
                rec['recovery_rounds'] = (recovered['round'].iloc[0] - tau) if len(recovered) else math.nan

        # —— E1：R1 恒等式（含 §11 同轮多客户端合成）——
        #   Ω_post = [n_pre·Ω_pre + Σ_j D_j² − D_sum²/n] / n，D_sum = ‖Σ_j δ_j‖
        #   m=1 时退化为 (n−1)/n·Ω_pre + (n−1)/n²·D²。多客户端需要 D_sum
        #  （交叉项无法由各 D_j 标量还原）；旧数据无 D_sum 的多客户端事件跳过。
        if len(ev):
            ji = ev[ev['event'] == 'join_identity']
            residuals = []
            for _, e in ji.iterrows():
                n = e.get('n_post')
                d_k = e.get('D_k') or {}
                if not n or e.get('omega_post') in (None, 0) or not d_k:
                    continue
                m = len(d_k)
                d_sum = e.get('D_sum')
                if m == 1:
                    d_sum2 = list(d_k.values())[0] ** 2 if d_sum is None else d_sum ** 2
                elif d_sum is not None and not pd.isna(d_sum):
                    d_sum2 = d_sum ** 2
                else:
                    continue  # 旧日志缺 D_sum：无法核对多客户端事件
                predicted = ((n - m) * e['omega_pre']
                             + sum(v ** 2 for v in d_k.values()) - d_sum2 / n) / n
                residuals.append(abs(e['omega_post'] - predicted) / max(e['omega_post'], 1e-30))
            if residuals:
                rec['e1_residual'] = max(residuals)

            # —— E2：τ_k 轮全网 η̂ 逐比特一致（去重计数应为 1）——
            sw = ev[ev['event'] == 'wc_eta_switch']
            if len(sw):
                rec['e2_eta_nunique'] = int(sw.groupby('tau_k')['eta'].nunique().max())

            wj = ev[ev['event'] == 'wc_join']
            if len(wj):
                rec['Delta_k'] = wj.iloc[0].get('Delta_k')
                rec['eta_hat'] = sw.iloc[0]['eta'] if len(sw) else None
        extra.append(rec)
    if not extra:
        return runs_df
    return runs_df.merge(pd.DataFrame(extra), on=['exp_group', 'run_name'], how='left')


def write_tables(runs_df, out_dir):
    metric_cols = [c for c in ('final_accuracy', 'M_window', 'acc_dip', 'recovery_rounds',
                               'Delta_k', 'eta_hat', 'wall_time_sec') if c in runs_df.columns]
    for group, gdf in runs_df.groupby('exp_group'):
        keys = ['dataset_name', 'arm'] if 'dataset_name' in gdf.columns else ['arm']
        if 'temp_client_dist' in gdf.columns and gdf['temp_client_dist'].nunique() > 1:
            keys.append('temp_client_dist')
        agg = gdf.groupby(keys)[metric_cols].agg(['mean', 'std', 'count'])
        agg.columns = ['_'.join(c) for c in agg.columns]
        agg = agg.round(6).reset_index()
        agg.to_csv(os.path.join(out_dir, f'table_{group}.csv'), index=False)
        try:
            with open(os.path.join(out_dir, f'table_{group}.md'), 'w', encoding='utf-8') as f:
                f.write(agg.to_markdown(index=False))
        except ImportError:
            pass  # to_markdown 需要 tabulate；缺失时只产出 CSV


def write_figures(rounds_df, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib 不可用，跳过出图')
        return
    for group, gdf in rounds_df.groupby('exp_group'):
        for scope, column, fname, logy in (('overall', 'accuracy', 'accuracy', False),
                                           ('network', 'Omega', 'omega', True)):
            sub = gdf[(gdf['scope'] == scope) & gdf.get(column).notna()] \
                if column in gdf.columns else pd.DataFrame()
            if not len(sub):
                continue
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for arm, adf in sub.groupby('arm'):
                curve = adf.groupby('round')[column].mean()
                ax.plot(curve.index, curve.values, label=arm, linewidth=1.2)
            if logy:
                ax.set_yscale('log')
            ax.set_xlabel('round')
            ax.set_ylabel(column)
            ax.set_title(group)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'fig_{group}_{fname}.png'), dpi=150)
            plt.close(fig)


def save_cache(df, path):
    try:
        df.to_parquet(path)
    except Exception:
        df.to_pickle(path.replace('.parquet', '.pkl'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--out', default=None, help='默认 <results_dir>/report')
    parser.add_argument('--groups', nargs='*', default=None, help='只聚合指定实验组')
    args = parser.parse_args()
    out_dir = args.out or os.path.join(args.results_dir, 'report')
    os.makedirs(out_dir, exist_ok=True)

    runs_df, rounds_df, events_df = scan_runs(args.results_dir, args.groups)
    if not len(runs_df):
        print(f'{args.results_dir} 下没有发现任何 run')
        return
    runs_df = derive_run_metrics(runs_df, rounds_df, events_df)

    save_cache(runs_df, os.path.join(out_dir, 'runs.parquet'))
    if len(rounds_df):
        save_cache(rounds_df, os.path.join(out_dir, 'rounds.parquet'))
    if len(events_df):
        save_cache(events_df, os.path.join(out_dir, 'events.parquet'))
    runs_df.to_csv(os.path.join(out_dir, 'runs_summary.csv'), index=False)

    write_tables(runs_df, out_dir)
    if len(rounds_df):
        write_figures(rounds_df, out_dir)

    # 验证类核对汇总（E1/E2）
    check_cols = [c for c in ('exp_group', 'run_name', 'arm', 'status',
                              'e1_residual', 'e2_eta_nunique') if c in runs_df.columns]
    checks = runs_df[check_cols]
    checks.to_csv(os.path.join(out_dir, 'checks.csv'), index=False)
    bad_e1 = checks[checks.get('e1_residual', pd.Series(dtype=float)) > 0.01] \
        if 'e1_residual' in checks.columns else pd.DataFrame()
    bad_e2 = checks[checks.get('e2_eta_nunique', pd.Series(dtype=float)) > 1] \
        if 'e2_eta_nunique' in checks.columns else pd.DataFrame()

    incomplete = runs_df[runs_df['status'] != 'COMPLETED']
    print(f"聚合完成：{len(runs_df)} runs（未完成 {len(incomplete)}）→ {out_dir}")
    if len(bad_e1):
        print(f"⚠ E1 恒等式残差 >1% 的 run：{list(bad_e1['run_name'])}")
    if len(bad_e2):
        print(f"⚠ E2 η̂ 不一致的 run：{list(bad_e2['run_name'])}")


if __name__ == '__main__':
    main()
