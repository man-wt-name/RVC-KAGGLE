# rvc_cli/preprocess.py
import os
import logging
import numpy as np
from rvc_cli.helpers import run_process # Импортируем helper

logger = logging.getLogger(__name__)
now_dir = os.getcwd()

# Перенесенный словарь sr_dict или импорт его из другого места
sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def preprocess_dataset_cli(trainset_dir, exp_dir_name, sr_str, n_p, config):
    """CLI version of preprocess_dataset."""
    if sr_str not in sr_dict:
        logger.error(f"Invalid sample rate: {sr_str}. Choose from {list(sr_dict.keys())}")
        return False
    sr = sr_dict[sr_str]
    log_dir = os.path.join(now_dir, "logs", exp_dir_name)
    os.makedirs(log_dir, exist_ok=True)

    # Используем config.python_cmd через helper
    cmd = f'python infer/modules/train/preprocess.py "{trainset_dir}" {sr} {n_p} "{log_dir}" {config.noparallel} {config.preprocess_per:.1f}'

    # Используем helper для запуска
    return run_process(cmd, config) == 0