"""run 级实验记录器。

目录结构（每个 run 自包含、append-only、崩溃安全）：
    results/<exp_group>/<run_name>/
        config.json     全部 args + 派生量（git commit、延迟时间表、λ̂ 等）
        metrics.jsonl   每轮指标：scope ∈ {overall, client, network}，每行一个 JSON
        events.jsonl    协议事件流：JOIN、η̂ 切换、加入恒等式、拓扑刷新等
        summary.json    结束时写一次：窗口平稳性 M、最终指标、墙钟、状态

聚合分析见 analysis/aggregate.py。
"""
import json
import os
import subprocess
from datetime import datetime


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _sanitize(value):
    """递归转成 JSON 可序列化：torch/numpy 标量与张量 → python 数值/列表。"""
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'item') and getattr(value, 'numel', lambda: 2)() == 1:
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class RunLogger:
    def __init__(self, results_dir, exp_group, run_name, args_dict, device=None):
        self.run_dir = os.path.join(results_dir, exp_group, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, 'metrics.jsonl')
        self.events_path = os.path.join(self.run_dir, 'events.jsonl')
        self.summary_path = os.path.join(self.run_dir, 'summary.json')
        self.config_path = os.path.join(self.run_dir, 'config.json')
        # 重跑未完成的 run：清掉旧的部分数据，保证文件内容与本次 run 一致
        for path in (self.metrics_path, self.events_path, self.summary_path):
            if os.path.exists(path):
                os.remove(path)

        self.config = {
            'exp_group': exp_group,
            'run_name': run_name,
            'status': 'RUNNING',
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'git_commit': _git_commit(),
            'compute_device': device,
            'args': _sanitize(args_dict),
        }
        self._write_config()

    def _write_config(self):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _append(self, path, record):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(_sanitize(record), ensure_ascii=False) + '\n')

    def log_metrics(self, round_num, scope, payload, client_id=None):
        record = {'round': round_num, 'scope': scope}
        if client_id is not None:
            record['client_id'] = client_id
        record.update(payload)
        self._append(self.metrics_path, record)

    def log_event(self, event_type, payload, round_num=None):
        record = {'event': event_type, 'round': round_num}
        record.update(payload)
        self._append(self.events_path, record)

    def save_config(self, key, value):
        self.config[key] = _sanitize(value)
        self._write_config()

    def finalize(self, summary=None, status='COMPLETED'):
        record = {'status': status,
                  'finished_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        record.update(_sanitize(summary or {}))
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        self.config['status'] = status
        self._write_config()

    @staticmethod
    def is_completed(results_dir, exp_group, run_name):
        path = os.path.join(results_dir, exp_group, run_name, 'summary.json')
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f).get('status') == 'COMPLETED'
        except Exception:
            return False
