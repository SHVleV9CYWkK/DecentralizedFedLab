"""极简事件总线：客户端 / Coordinator 向 RunLogger 上报结构化协议事件。

单进程仿真（n_job=1）专用。未 attach 时所有调用为 no-op，
因此客户端代码可以无条件 emit，冒烟测试与旧入口不受影响。
"""

_logger = None
_current_round = None


def attach(logger):
    global _logger
    _logger = logger


def detach():
    global _logger
    _logger = None


def set_round(round_num):
    global _current_round
    _current_round = round_num


def emit(event_type, **payload):
    if _logger is not None:
        _logger.log_event(event_type, payload, round_num=_current_round)
