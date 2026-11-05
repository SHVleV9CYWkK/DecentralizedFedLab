"""论文用图导出（标签与正文严格一致）。

用法：
    python -m analysis.paper_figures --report results/report --out ../paper/figures

与 `analysis.aggregate` 的 `write_figures` 的区别：
  * 这里导出的是**论文正文引用的图**，矢量 PDF，字号/尺寸按 ICLR 单栏排版设定；
  * 曲线与臂的标签一律使用**论文里的名字**（``Init + calib.`` / ``Calib. only`` /
    ``Init only`` / ``D-PSGD``），不使用代码里的 ``wc`` / ``w_only`` / ``c_only`` /
    ``cold`` 标识符。两套名字的映射见 ``ARM_LABELS``；改论文里的臂名时改这里，
    不要在 LaTeX 里手工改图。

当前导出：
  fig1_ushape.pdf   Sec.5「校准步长」小节的图 1。M(η) 对 log η，左稀疏图
                    （cap-active 分支）、右完全图（off-cap 分支），标出网格最优
                    η_best、校准值 η̂、以及上限 η_c；观察性准确率画在次纵轴上，
                    以显示 M-最优步长与 acc-最优步长不重合（GR2）。
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

# ---------------------------------------------------------------- 命名映射
# 论文里的臂名（sections/05_experiments.tex）。左边是代码/日志里的 arm 标识符。
ARM_LABELS = {
    'wc': 'Init + calib.',
    'wc[calibrate=0]': 'Init only',
    'wc[warm_mode=cold]': 'Calib. only',
    'wc[warm_mode=cold,calibrate=0]': 'D-PSGD',
    'dfedavg': 'DFedAvg',
    'ellocal': 'Epidemic Learning',
    'localonly': 'No collaboration',
}

# 论文里的指标名。
M_LABEL = r'certified metric $M$'
ACC_LABEL = 'accuracy (obs.)'
ETA_LABEL = r'step size $\eta$'


def paper_arm_label(arm: str) -> str:
    """把日志里的 arm 标识符翻译成论文里的名字；未登记的原样返回。"""
    return ARM_LABELS.get(arm, arm)


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 7.5,
        'axes.labelsize': 7.5,
        'axes.titlesize': 8,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    return plt


def _grid_frame(runs: pd.DataFrame, group: str) -> pd.DataFrame:
    """取出一个步长网格组，返回按 η 排序的 (eta, M, acc) 表，并标出校准臂。

    网格臂的 arm 形如 ``wc[eta_frac=0.125,...]``；同组里不带 ``eta_frac`` 的那一
    条就是校准臂（η̂ 由算法自己算出来）。
    """
    g = runs[runs['exp_group'] == group]
    rows = []
    for arm, adf in g.groupby('arm'):
        m = re.search(r'eta_frac=([0-9.]+)', arm)
        rows.append({
            'arm': arm,
            'is_grid': m is not None,
            'eta_frac': float(m.group(1)) if m else None,
            'eta': adf['eta_hat'].mean(),
            'M': adf['M_window'].mean(),
            'M_std': adf['M_window'].std(),
            'acc': adf['final_accuracy'].mean(),
        })
    df = pd.DataFrame(rows).sort_values('eta').reset_index(drop=True)
    return df


def fig1_ushape(runs: pd.DataFrame, out_dir: str,
                sparse_group: str = 'G2_grid',
                full_group: str = 'G9_grid_full') -> str:
    """图 1：M(η) 的 U 形与校准落点，两个拓扑分支各一栏。"""
    plt = _mpl()
    panels = [
        (sparse_group, r'sparse graph ($\hat\lambda\approx0.88$), cap active'),
        (full_group, r'complete graph ($\hat\lambda=0.50$), cap inactive'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.55))

    for ax, (group, title) in zip(axes, panels):
        df = _grid_frame(runs, group)
        grid = df[df.is_grid]
        calib = df[~df.is_grid]
        if not len(grid):
            continue
        # 主轴：认证指标 M（越低越好）
        ax.plot(grid['eta'], grid['M'], 'o-', color='#1f4e79', markersize=3.2,
                linewidth=1.1, label=M_LABEL, zorder=3)
        ax.fill_between(grid['eta'], grid['M'] - grid['M_std'],
                        grid['M'] + grid['M_std'], color='#1f4e79',
                        alpha=0.12, linewidth=0, zorder=1)

        best = grid.loc[grid['M'].idxmin()]
        ax.scatter([best['eta']], [best['M']], marker='v', s=34,
                   color='#1f4e79', zorder=5)
        ax.annotate(r'$\eta_{\mathrm{best}}$', (best['eta'], best['M']),
                    textcoords='offset points', xytext=(0, -12), ha='center',
                    fontsize=7, color='#1f4e79')

        eta_c = grid['eta'].max()          # 网格上端 η_frac=1 即上限 η_c
        c = calib.iloc[0] if len(calib) else None
        # cap-active 分支上 η̂ 与 η_c 重合，画两条线只会挤成一团：合并成一条。
        capped = c is not None and abs(c['eta'] / eta_c - 1.0) < 0.02

        if not capped:
            ax.axvline(eta_c, color='0.45', linestyle=':', linewidth=0.9, zorder=2)
            ax.annotate(r'$\eta_{\mathrm{c}}$', (eta_c, 1.0),
                        xycoords=('data', 'axes fraction'),
                        textcoords='offset points', xytext=(-2, -9), ha='right',
                        fontsize=7, color='0.35')

        if c is not None:
            ax.axvline(c['eta'], color='#b03a2e', linestyle='--', linewidth=0.9, zorder=2)
            ax.scatter([c['eta']], [c['M']], marker='*', s=70, color='#b03a2e', zorder=6)
            ax.annotate(r'$\hat\eta=\eta_{\mathrm{c}}$' if capped else r'$\hat\eta$',
                        (c['eta'], 1.0), xycoords=('data', 'axes fraction'),
                        textcoords='offset points', xytext=(-2, -9), ha='right',
                        fontsize=7, color='#b03a2e')

        ax.set_xscale('log')
        ax.set_xlabel(ETA_LABEL, labelpad=1.5)
        ax.set_ylabel(M_LABEL, labelpad=1.5)
        ax.set_title(title, pad=3)
        ax.tick_params(direction='in', top=False, right=False)
        # 下方留够空间给 η_best 的标注，上方留给 η̂ / η_c 的标注。
        lo, hi = grid['M'].min(), grid['M'].max()
        ax.set_ylim(lo - 0.30 * (hi - lo), hi + 0.34 * (hi - lo))

        # 次轴：观察性准确率（GR2：理论不对其作承诺）
        ax2 = ax.twinx()
        ax2.plot(grid['eta'], grid['acc'], 's--', color='#7f8c8d', markersize=2.6,
                 linewidth=0.9, label=ACC_LABEL, zorder=2)
        ax2.set_ylabel(ACC_LABEL, color='#5d6d7e', labelpad=1.5)
        ax2.tick_params(axis='y', colors='#5d6d7e', direction='in')
        ax2.spines['right'].set_color('#95a5a6')
        ax2.margins(y=0.16)

    handles = [
        plt.Line2D([], [], color='#1f4e79', marker='o', markersize=3.2,
                   linewidth=1.1, label=M_LABEL),
        plt.Line2D([], [], color='#7f8c8d', marker='s', markersize=2.6,
                   linewidth=0.9, linestyle='--', label=ACC_LABEL),
        plt.Line2D([], [], color='#b03a2e', linestyle='--', linewidth=0.9,
                   label=r'calibrated $\hat\eta$'),
        plt.Line2D([], [], color='0.45', linestyle=':', linewidth=0.9,
                   label=r'cap $\eta_{\mathrm{c}}=(1-\lambda_n)/(7L)$'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.3, handlelength=1.7,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.6)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'fig1_ushape.pdf')
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='results/report',
                    help='analysis.aggregate 的输出目录（含 runs.parquet）')
    ap.add_argument('--out', default='../paper/figures',
                    help='PDF 输出目录（论文工程的 figures/）')
    args = ap.parse_args()

    runs = pd.read_parquet(os.path.join(args.report, 'runs.parquet'))
    print('wrote', fig1_ushape(runs, args.out))


if __name__ == '__main__':
    main()
