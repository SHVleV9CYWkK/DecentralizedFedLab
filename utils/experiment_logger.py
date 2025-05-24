import os
import json
import random
from datetime import datetime
from contextlib import ContextDecorator

class ExperimentLogger(ContextDecorator):
    def __init__(self, today_date, experiment_num, device, args):
        self.args = vars(args)
        self.save_log_dir = args.log_dir
        self.dataset_name = args.dataset_name
        self.fl_type = args.fl_method
        self.seed = args.seed
        self.device_type = device.type

        if self.seed is not None:
            random.seed(self.seed)

        self.today_date = today_date
        self.experiment_num = experiment_num

        self.log_dir = os.path.join(
            self.save_log_dir,
            self.today_date,
            self.dataset_name,
            self.fl_type,
            self.experiment_num
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.config_data = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "experiment_num": self.experiment_num,
            "status": "RUNNING",
            "compute_device": self.device_type,
            "args": self.args
        }

        self.config_file_path = os.path.join(self.log_dir, "experiment_config.json")
        self._write_config()

    def _write_config(self):
        with open(self.config_file_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, indent=2, ensure_ascii=False)

    def save(self, save_name, value):
        self.config_data[save_name] = value
        self._write_config()

    def update_status(self, status):
        self.config_data["status"] = status
        self._write_config()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.update_status("COMPLETED")
        else:
            self.update_status("INTERRUPTED")
        return False