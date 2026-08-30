import math
import random
from collections import deque
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Subset

from clients.client import Client
from utils import event_bus


# W+C 方法论：热启动(W) + 失配校准步长(C)
# 依据《WC方法论_实现规范_v1.0》（组件 H 按规范决策排除）：
#   - 组件 W（§4, W-N 变体）：加入前一轮 Coordinator 已把邻居模型推送给新客户端
#     （pre_add 机制 = PULL#2 新鲜快照），set_init_model 时取邻居平均作 θ_warm，
#     把加入位移 D_k 压到共识尺度，消除冲击项 (ii)。【R1 / Cor. warm】
#   - 组件 C（§5）：新客户端本地估计失配 Δ̂_k（Algorithm 7），按
#     η̂ = min{(1−λ̂)/(7L̂), sqrt(2Ĝ_n/(L̂σ̂²T'))} 校准（Algorithm 8），
#     τ_k 轮全网（含在位者）同步切换到常数 η̂，无 warmup/衰减/重启。【R4/R6】
#   - 网络常数 L̂/σ̂²（Algorithm 3）与平台位 P（Algorithm 4）在 Phase-1 由各客户端
#     周期维护，并随模型交换捎带洪泛（max / AND 聚合）。
#   - 度数与 λ̂_post 由 Coordinator 经 set_topology_info 下发：等价于规范的
#     DEGREE 交换（§6.5）与研究模式 λ̂ 精确特征值分解（§5.3 [T 精确]）。
#
# 与规范的仿真侧偏差（诚实声明，均为规范允许的研究模式/[H] 路径）：
#   1. JOIN 洪泛用类级公告板模拟（§5.5 要求消息先于 τ_k 送达全网；单进程仿真中
#      公告板等价于按时送达的洪泛）。因此要求 --n_job=1（多进程下无共享内存）。
#   2. 平台位 P 的 AND 聚合为一跳近似 [H]；稠密图下即全局 AND。
#   3. 非平台加入无法推迟（加入轮由 Coordinator 指定），采用 §5.2 回退 (b)：
#      ε̂₁ 取在位者 (EMA损失 − 历史最低EMA) 的邻居平均 [H]。
#   4. 本框架每轮做多个本地 SGD 步（local_epochs × 批数），校准公式中的 T' 取
#      剩余 SGD 总步数（每步即一次步长为 η 的随机梯度步）；估计误差由 R5 的
#      (√κ+1/√κ)/2 平方根级鲁棒性覆盖。
#   5. 式 (U) 为纯 SGD 半步，故主循环强制 momentum=0 的 SGD（动量会使有效步长
#      偏离校准值）；Algorithm 7 的本地拟合用 Adam（其权重最终丢弃，R-W1）。
#
# 消融开关（实验清单 E4/E5/E6/E8/E9/E14 专用，默认全部关闭=规范默认行为）：
#   wc_warm_mode   : neighbor(默认) / global_sim(W-G) / cold(冷加入) / fitted(故意违反 R-W1)
#   wc_calibrate   : 0 时跳过校准（"仅W"臂）
#   wc_post_schedule: constant(默认) / sqrt_decay / cosine / warmup（E9 调度臂 [H]）
#   wc_eta_frac    : >0 时 η̂ := frac·η_c（E6 网格 {η_c·2^-j}）
#   wc_kappa_g     : Ĝ_n 乘性扰动（E8 误标定鲁棒性）
#   lambda_hat_override: >0 时强制 λ̂（E14 高/低估方向）
class WCClient(Client):

    # 类级公告板：{tau_k: [JOIN事件, ...]}，模拟 JOIN 洪泛（见上方声明 1）
    _join_board = {}

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        if client_id == 0:
            WCClient._join_board.clear()

        h = hyperparam
        self.bz = h['bz']
        self.n_clients = h.get('n_clients', 1)          # 仅 ε̂₁>0 与无噪声分支需要
        self.lambda_hat = h.get('lambda_hat', 0.9)      # λ̂ 保守上界（部署模式回退，§5.3）
        # Algorithm 3：常数估计
        self.c_L = float(h.get('c_L', 2.0))             # L̂ 保守系数（不可 <1）
        self.K_const = h.get('K_const', 5)
        self.n_probe_L = h.get('n_probe_L', 4)
        self.delta_rel = h.get('delta_rel', 1e-2)
        self.B_sigma = h.get('B_sigma', 8)
        self.batch_L = h.get('batch_L', max(512, 8 * self.bz))
        # Algorithm 4：平台检测
        self.plateau_alpha = h.get('plateau_alpha', 0.3)
        self.W_p = h.get('W_p', 5)
        self.tol_p = h.get('tol_p', 1e-2)
        # Algorithm 7：失配估计
        self.m_loc = h.get('m_loc', 200)
        self.e_eval = h.get('e_eval', 20)
        self.patience = h.get('patience', 5)
        self.min_delta = h.get('min_delta', 1e-4)
        n_eval = min(self.train_dataset_len, h.get('n_eval', 4096))
        # Algorithm 8：校准
        self.sigma2_min = h.get('sigma2_min', 1e-12)
        self.zeta2 = h.get('zeta2', 1.0)                # 仅无噪声分支需要
        self.eta_min_frac = h.get('eta_min_frac', 0.0)  # [H] 护栏，默认关闭
        # 消融开关
        self.wc_warm_mode = h.get('wc_warm_mode', 'neighbor')
        self.wc_calibrate = bool(int(h.get('wc_calibrate', 1)))
        self.wc_post_schedule = h.get('wc_post_schedule', 'constant')
        self.wc_eta_frac = float(h.get('wc_eta_frac', 0.0))
        self.wc_kappa_g = float(h.get('wc_kappa_g', 1.0))
        self.lambda_hat_override = float(h.get('lambda_hat_override', -1.0))
        self.wc_force_eps1_zero = bool(int(h.get('wc_force_eps1_zero', 0)))

        # 固定评测集 E_k（§5.1-1：全程复用同一 E_k）——用未增强的原始数据，
        # 保证 ℓ_warm 与 f̂_k^loc 的确定性可比（增强只进训练目标）
        eval_base = getattr(self, 'train_eval_dataset', self.client_train_loader.dataset)
        eval_subset = Subset(eval_base, list(range(n_eval)))
        self._eval_loader = DataLoader(eval_subset, batch_size=self.bz, shuffle=False)

        # 持久状态
        self.t = 0                       # 本地同步轮计数（train 结束自增）
        self.L_loc = self.L_net = 0.0    # L̂：本地 / 网络 max
        self.s2_loc = self.s2_net = 0.0  # σ̂²：本地 / 网络 max
        self.p_local = False             # 平台本地位
        self.P_net = False               # 平台网络 AND 位
        self.ema = None
        self.ema_min = float('inf')
        self.ema_hist = deque(maxlen=self.W_p + 1)
        self.last_degree = 0             # 入度回退值（Coordinator 未下发度数时使用）
        self.my_degree = None            # Coordinator 下发的真实活跃度数（DEGREE 语义）
        self.lambda_exact = None         # Coordinator 下发的精确 λ̂（研究模式）
        self.n_active = None             # 当前活跃客户端数（§11 顺序加入的 n_post）
        self._post_sched = None          # 校准后调度状态（E9 调度臂）
        self._last_g_n = None            # 最近一次校准用的 Ĝ_n（落盘诊断）
        self._applied_events = set()     # 已生效的 JOIN（按 tau_k）

    # ---------------- 框架接口 ----------------

    def init_client(self):
        # 式 (U) 为纯 SGD 半步；动量会使有效步长偏离校准值（声明 5），
        # 故不走 get_optimizer，强制无动量 SGD + 常数步长（R6：无任何调度）
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = None

    def update_lr(self):
        # R6：常数步长在该界类中最优，禁用一切外部 lr 调度
        pass

    def set_topology_info(self, degree, lam, n_active):
        """Coordinator 下发：活跃图真实度数 + 精确 λ̂（DEGREE 交换 / 研究模式 §5.3）。"""
        self.my_degree = degree
        if lam is not None:
            self.lambda_exact = lam
        self.n_active = n_active

    def set_init_model(self, model):
        self.model = deepcopy(model)
        if len(self.neighbor_model_weights) != 0:
            # pre_add 轮收到的邻居模型 = PULL#2 新鲜快照 → 加入协议（J4/J2/J3）
            self._join_warm_start()

    def train(self):
        self._apply_pending_join()       # τ_k 轮全网同步切换 η̂（§5.5，含在位者）
        self._maybe_schedule_lr()        # E9 调度臂（默认 constant 时为 no-op）
        self.model.train()
        round_losses = []
        for _ in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), labels).mean()
                loss.backward()
                self.optimizer.step()
                round_losses.append(loss.item())
        # 统计维护（Algorithm 1 第(5)步）
        if round_losses:
            self._plateau_update(sum(round_losses) / len(round_losses))
        if self.t % self.K_const == 0:
            self._refresh_constants()
        self.t += 1

    def send_model(self):
        if self.model is None:
            return None
        # 快照由 Coordinator 的 _snapshot 统一完成（同步轮语义），此处只组装内容
        meta = {
            'degree': self.my_degree if self.my_degree is not None else self.last_degree,
            'L_net': self.L_net,
            'sigma2_net': self.s2_net,
            'p_local': self.p_local,
            'P_net': self.P_net,
            't_now': self.t,
            'T_total': self.n_rounds,
            'eps1_proxy': self._eps1_proxy(),
        }
        return self.model.state_dict(), meta

    def aggregate(self):
        payloads = [p for p in self.neighbor_model_weights if p is not None]
        if len(payloads) == 0:
            self.neighbor_model_weights.clear()
            return
        sds = [p[0] for p in payloads]
        metas = [p[1] for p in payloads]
        d_i = len(sds)
        self.last_degree = d_i

        # SCALAR 捎带洪泛吸收：L̂/σ̂² 取 max，平台位取 AND（声明 2）
        for meta in metas:
            self.L_net = max(self.L_net, meta['L_net'])
            self.s2_net = max(self.s2_net, meta['sigma2_net'])
        self.P_net = self.p_local and all(m['p_local'] for m in metas)

        # Algorithm 2：lazy Metropolis 混合 W=(I+M)/2；度数来自 Coordinator 下发（精确）
        my_deg = self.my_degree if self.my_degree is not None else d_i
        w_nbr = [0.5 / (1.0 + max(my_deg, meta['degree'])) for meta in metas]
        w_self = 1.0 - sum(w_nbr)
        own = self.model.state_dict()
        mixed = {}
        for key, v in own.items():
            if v.is_floating_point():
                acc = v * w_self
                for w, sd in zip(w_nbr, sds):
                    acc = acc + sd[key].to(self.device) * w
                mixed[key] = acc
            else:
                mixed[key] = v
        self.model.load_state_dict(mixed)
        self.neighbor_model_weights.clear()

    # ---------------- 加入协议（新客户端侧，Algorithm 5） ----------------

    def _join_warm_start(self):
        payloads = [p for p in self.neighbor_model_weights if p is not None]
        self.neighbor_model_weights.clear()
        if len(payloads) == 0:
            print(f"[WC] client {self.id}: 加入轮未收到任何邻居模型，退化为冷加入（范围声明 §4.3）")
            return
        sds = [p[0] for p in payloads]
        metas = [p[1] for p in payloads]

        # —— 吸收元数据（JOIN 钉死元组的来源，标量取保守方向）——
        L_hat = max(m['L_net'] for m in metas)
        s2_hat = max(m['sigma2_net'] for m in metas)
        P = all(m['P_net'] for m in metas)
        tau_k = max(m['t_now'] for m in metas)
        self.t = tau_k
        self.last_degree = len(sds)
        self.L_net, self.s2_net = L_hat, s2_hat

        # —— 组件 W：θ_warm = 收到模型的平均 ——
        # neighbor 模式来源是图邻居（W-N）；global_sim 模式 Coordinator 给全部活跃
        # 客户端的模型（W-G，§4.2）。R-W1：θ_k^{τ_k} 只能取此平均。
        theta_warm = {}
        for key, v in self.model.state_dict().items():
            if v.is_floating_point():
                theta_warm[key] = sum(sd[key].to(self.device) for sd in sds) / len(sds)
            else:
                theta_warm[key] = sds[0][key].to(self.device)
        if self.wc_warm_mode != 'cold':
            self.model.load_state_dict(theta_warm)

        # —— 平台门（§5.2）：P=1 → ε̂₁:=0（n 消去）；否则回退 (b)（声明 3）——
        # gate_path 记录实际走的分支：G_n 的 n 依赖只从回退项 (n−1)·ε̂₁ 进入，
        # 因此"n 消去"只在 plateau/forced 分支上成立，必须逐 run 落盘可分辨。
        if self.wc_force_eps1_zero:
            eps1, gate_path = 0.0, 'forced'
        elif P:
            eps1, gate_path = 0.0, 'plateau'
        else:
            eps1, gate_path = sum(m['eps1_proxy'] for m in metas) / len(metas), 'fallback'

        # —— J2：本地失配估计 Δ̂_k（Algorithm 7）——
        delta_k = None
        if self.wc_warm_mode == 'fitted':
            # E5 消融臂：故意违反 R-W1，用拟合权重作 θ_k^{τ_k} → 冲击项按 R1 精确复活
            delta_k, fit_state = self._estimate_misfit(return_fit_state=True)
            self.model.load_state_dict(fit_state)
        elif self.wc_calibrate:
            delta_k = self._estimate_misfit()

        lam_post = self._lam_post()
        n_post = self.n_active if self.n_active is not None else self.n_clients
        T_total = self.n_rounds
        steps_per_round = max(1, self.epochs * len(self.client_train_loader))
        event_bus.emit('wc_join', k_id=self.id, tau_k=tau_k, Delta_k=delta_k,
                       eps1=eps1, P=P, L=L_hat, sigma2=s2_hat, lam_post=lam_post,
                       n_post=n_post, warm_mode=self.wc_warm_mode,
                       calibrate=self.wc_calibrate, gate_path=gate_path,
                       n_neighbors=len(sds), shard_size=self.train_dataset_len,
                       local_steps=steps_per_round)

        if not self.wc_calibrate:
            return
        if L_hat <= 0.0 or tau_k >= T_total:
            print(f"[WC] client {self.id}: 校准条件不满足 (L̂={L_hat:.3g}, tau_k={tau_k}, T={T_total})，"
                  f"保持默认步长 {self.lr}")
            return

        event = {
            'k_id': self.id, 'tau_k': tau_k, 'T_total': T_total,
            'Delta_k': delta_k, 'eps1': eps1, 'P': P,
            'L': L_hat, 'sigma2': s2_hat, 'lam_post': lam_post,
            'n_post': n_post, 'steps_per_round': steps_per_round,
            'gate_path': gate_path,
        }
        # —— J3：洪泛 JOIN（公告板模拟，声明 1）；τ_k 轮所有客户端在 train() 入口生效 ——
        WCClient._join_board.setdefault(tau_k, []).append(event)
        print(f"[WC] JOIN: client {self.id} tau_k={tau_k} Delta_k={delta_k:.4f} "
              f"eps1={eps1:.4f} P={P} L={L_hat:.4f} sigma2={s2_hat:.4f} lam={lam_post:.4f}")

    def _lam_post(self):
        if self.lambda_hat_override > 0:
            return self.lambda_hat_override   # E14 扰动臂：强制高/低估方向
        if self.lambda_exact is not None:
            return self.lambda_exact          # 研究模式：Coordinator 精确特征值 [T]
        return self.lambda_hat                # 部署模式：保守常数 [H]

    # ---------------- Algorithm 7：本地失配估计 Δ̂_k ----------------

    def _estimate_misfit(self, return_fit_state=False):
        # 损失口径 = 训练目标（CE 均值，无额外正则），评测一律 eval 模式（§5.1-2）
        l_warm = self._eval_loss(self.model)
        # 在深拷贝上拟合：权重随副本丢弃，θ_k^{τ_k} 保持 θ_warm（R-W1）
        fit_model = deepcopy(self.model)
        opt = torch.optim.Adam(fit_model.parameters(), lr=1e-3)
        best, since, s = l_warm, 0, 0
        done = False
        while s < self.m_loc and not done:
            for x, labels in self.client_train_loader:
                fit_model.train()
                x, labels = x.to(self.device), labels.to(self.device)
                opt.zero_grad()
                loss = self.criterion(fit_model(x), labels).mean()
                loss.backward()
                opt.step()
                s += 1
                if s % self.e_eval == 0:
                    l = self._eval_loss(fit_model)
                    if l < best - self.min_delta:
                        best, since = l, 0
                    else:
                        since += 1
                    if since >= self.patience:
                        done = True
                        break
                if s >= self.m_loc:
                    break
        # f̂_k^loc = best ≥ f_k^* → Δ̂ 轻度低估，双向均被 R5 覆盖（§5.1-4）
        delta = max(0.0, l_warm - best)
        if return_fit_state:
            return delta, {k: v.detach().clone() for k, v in fit_model.state_dict().items()}
        return delta

    def _eval_loss(self, model):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x, labels in self._eval_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                losses = self.criterion(model(x), labels)
                total += float(losses.sum())
                count += labels.numel()
        return total / max(count, 1)

    # ---------------- Algorithm 8：步长校准（全网确定性同值） ----------------

    def _apply_pending_join(self):
        for tau in sorted(WCClient._join_board.keys()):
            if tau <= self.t and tau not in self._applied_events:
                events = WCClient._join_board[tau]
                verbose = any(e['k_id'] == self.id for e in events)
                eta, eta_c = self._calibrate_eta(events, verbose=verbose)
                self.lr = eta
                for group in self.optimizer.param_groups:
                    group['lr'] = eta
                self._applied_events.add(tau)
                self._post_sched = {'kind': self.wc_post_schedule, 'eta_hat': eta,
                                    'eta_c': eta_c, 'tau': tau,
                                    'T_total': events[0]['T_total']}
                # 每个客户端都上报切换值 → 全网一致性核对；同时落盘校准的全部输入，
                # 使 η̂ 的 n 依赖（经 G_n 的回退项）可以逐 run 复算
                event_bus.emit('wc_eta_switch', client_id=self.id, tau_k=tau, eta=eta,
                               eta_c=eta_c, cap_active=bool(eta >= eta_c - 1e-15),
                               gate_path=events[0].get('gate_path'),
                               G_n=self._last_g_n, n_post=events[0]['n_post'])
                if verbose:
                    print(f"[WC] switch: 全网于 t={tau} 同步切换到常数步长 eta_hat={eta:.3e}")

    def _calibrate_eta(self, events, verbose=False):
        # 输入仅来自 JOIN 钉死的事件元组 + 静态配置 → 所有节点输出逐比特相同（§3.4）
        L = max(e['L'] for e in events)
        s2 = max(e['sigma2'] for e in events)
        lam = max(e['lam_post'] for e in events)
        eps1 = max(e['eps1'] for e in events)
        tau_k = events[0]['tau_k']
        T_total = events[0]['T_total']
        n_post = max(e['n_post'] for e in events)
        spr = max(e['steps_per_round'] for e in events)
        m = len(events)
        # 同轮并发到达（§11）：Ĝ_n = Σ_j Δ̂_kj + (n_post − m)·ε̂₁；平台门下 ε̂₁=0 → n 消去
        g_n = sum(e['Delta_k'] for e in events) + max(0, n_post - m) * eps1
        g_n *= self.wc_kappa_g                          # E8 κ 扰动（静态配置，全网一致）
        self._last_g_n = g_n                            # 落盘用：n 依赖只从 (n−m)·ε̂₁ 进入

        eta_c = (1.0 - lam) / (7.0 * L)                 # 封顶 η_c，不可逾越 [T]
        if self.wc_eta_frac > 0:                        # E6 网格臂：η̂ := frac·η_c（规范 V2）
            return self.wc_eta_frac * eta_c, eta_c
        t_rem = (T_total - tau_k) * spr                 # T' 取剩余 SGD 总步数（声明 4）
        if s2 > self.sigma2_min:                        # 主分支（R4）
            eta_raw = math.sqrt(2.0 * g_n / (L * s2 * t_rem)) if g_n > 0 else 0.0
            if t_rem < 98.0 * L * g_n / (s2 * (1.0 - lam) ** 2 + 1e-30):
                if verbose:
                    print(f"[WC] cap_active: tau_k={tau_k}（极晚加入，η̂ 触顶 η_c，R4 阈值）")
                    event_bus.emit('wc_cap_active', tau_k=tau_k)
        else:                                           # 近无噪声分支（R8，立方根律）
            eta_raw = (g_n * (1.0 - lam) ** 2
                       / (48.0 * n_post * t_rem * L ** 2 * self.zeta2)) ** (1.0 / 3.0)
        eta = min(eta_c, eta_raw)
        if self.eta_min_frac > 0:                       # [H] 护栏，默认关闭
            eta = max(eta, self.eta_min_frac * eta_c)
        if eta <= 0.0 and verbose:
            print("[WC] warning: Ĝ_n≈0 → η̂=0（界视角无害，见 §8-11；可用 eta_min_frac 护栏）")
        return eta, eta_c

    def _maybe_schedule_lr(self):
        """E9 调度消融臂 [H]：在校准 η̂ 的基础上施加 warmup/衰减形状（全部封顶 η_c）。

        规范默认（constant）下为 no-op——R6：已知地平线下调度形状不是杠杆。
        """
        s = self._post_sched
        if s is None or s['kind'] == 'constant':
            return
        t_rel = max(0, self.t - s['tau'])
        T_rel = max(1, s['T_total'] - s['tau'])
        eta_hat, eta_c = s['eta_hat'], s['eta_c']
        if s['kind'] == 'sqrt_decay':
            lr = eta_hat * math.sqrt(T_rel) / math.sqrt(t_rel + 1)
        elif s['kind'] == 'cosine':
            eta_min = 0.01 * eta_hat
            lr = eta_min + 0.5 * (eta_hat - eta_min) * (1 + math.cos(math.pi * min(1.0, t_rel / T_rel)))
        elif s['kind'] == 'warmup':
            ramp = max(1, T_rel // 10)
            lr = eta_hat * min(1.0, (t_rel + 1) / ramp)
        else:
            return
        lr = min(lr, eta_c)
        self.lr = lr
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    # ---------------- Algorithm 3：网络常数估计（L̂、σ̂²） ----------------

    def _refresh_constants(self):
        was_training = self.model.training
        self.model.eval()  # 关 dropout，探针确定性（§8-2）；探针不更新权重
        params = [p for p in self.model.parameters() if p.requires_grad]

        # L̂：方向探针，两次梯度必须用同一大批次以隔离曲率与噪声（§8-4）
        x, y = self._sample_batch(self.batch_L)
        g0 = self._flat_grad(x, y)
        theta_norm = math.sqrt(sum(float(p.detach().pow(2).sum()) for p in params))
        delta = self.delta_rel * max(1.0, theta_norm)
        l_max = 0.0
        for _ in range(self.n_probe_L):
            us = [torch.randn_like(p) for p in params]
            u_norm = math.sqrt(sum(float(u.pow(2).sum()) for u in us)) + 1e-12
            with torch.no_grad():
                for p, u in zip(params, us):
                    p.add_(u, alpha=delta / u_norm)
            g1 = self._flat_grad(x, y)  # 同一批次 ξ_L
            with torch.no_grad():
                for p, u in zip(params, us):
                    p.sub_(u, alpha=delta / u_norm)
            l_max = max(l_max, float((g1 - g0).norm()) / delta)
        self.L_loc = max(self.L_loc, self.c_L * l_max)  # 保守方向：宁可高估（§5.3）

        # σ̂²：同一参数点、B 个独立小批量的样本方差（§8-5）
        grads = [self._flat_grad(*self._sample_batch(self.bz)) for _ in range(self.B_sigma)]
        g_stack = torch.stack(grads)
        g_bar = g_stack.mean(dim=0)
        s2 = float((g_stack - g_bar).pow(2).sum() / (self.B_sigma - 1))
        self.s2_loc = max(self.s2_loc, s2)

        self.L_net = max(self.L_net, self.L_loc)
        self.s2_net = max(self.s2_net, self.s2_loc)
        if was_training:
            self.model.train()

    def _flat_grad(self, x, labels):
        self.model.zero_grad(set_to_none=True)
        loss = self.criterion(self.model(x), labels).mean()
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params)
        return torch.cat([g.reshape(-1) for g in grads]).detach()

    def _sample_batch(self, batch_size):
        dataset = self.client_train_loader.dataset
        idx = [random.randrange(len(dataset)) for _ in range(min(batch_size, len(dataset)))]
        samples = [dataset[i] for i in idx]
        xs = torch.stack([s[0] for s in samples]).to(self.device)
        ys = torch.as_tensor([int(s[1]) for s in samples]).to(self.device)
        return xs, ys

    # ---------------- Algorithm 4：平台期检测（ε̂₁ 门） ----------------

    def _plateau_update(self, loss_value):
        if self.ema is None:
            self.ema = loss_value
        else:
            self.ema = (1.0 - self.plateau_alpha) * self.ema + self.plateau_alpha * loss_value
        self.ema_min = min(self.ema_min, self.ema)
        self.ema_hist.append(self.ema)
        if len(self.ema_hist) > self.W_p:
            old = self.ema_hist[0]
            self.p_local = abs(self.ema - old) / max(abs(old), 1e-8) < self.tol_p
        # P_net 在 aggregate 中与邻居位求 AND；本地位变化后兜底收紧
        self.P_net = self.P_net and self.p_local

    def _eps1_proxy(self):
        # §5.2 非平台回退 (b)：ε̂₁ ≈ 当前 EMA 损失 − 历史最低 EMA [H]
        if self.ema is None or not math.isfinite(self.ema_min):
            return 0.0
        return max(0.0, self.ema - self.ema_min)
