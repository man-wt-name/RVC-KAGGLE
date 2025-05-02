# --- rvc_cli/extract.py ---
import os
import logging
import numpy as np
from rvc_cli.helpers import run_process, run_process_parallel # Импортируем helpers

logger = logging.getLogger(__name__)
now_dir = os.getcwd()

def extract_f0_feature_cli(gpus_str, n_p, f0method, if_f0, exp_dir_name, version, gpus_rmvpe_str, config):
    """CLI version of extract_f0_feature."""
    gpus_list = gpus_str.split("-") if gpus_str else []
    log_dir = os.path.join(now_dir, "logs", exp_dir_name)
    os.makedirs(log_dir, exist_ok=True)

    f0_success = True
    if if_f0:
        logger.info("Starting F0 extraction...")
        if f0method != "rmvpe_gpu":
             cmd = (
                f'python infer/modules/train/extract/extract_f0_print.py "{log_dir}" {n_p} {f0method}'
            )
             if run_process(cmd, config) != 0:
                 logger.error(f"F0 extraction ({f0method}) failed.")
                 f0_success = False
        else:
            gpus_rmvpe_list = gpus_rmvpe_str.split("-") if gpus_rmvpe_str else []
            if not gpus_rmvpe_list or gpus_rmvpe_list == ['']:
                 logger.warning("No GPUs specified for rmvpe_gpu. F0 extraction might be slow or fail.")
            elif gpus_rmvpe_list == ["-"]:
                 logger.warning("GPU usage explicitly disabled for rmvpe_gpu ('-'). Skipping GPU F0 extraction.")
            else:
                f0_cmds = []
                leng = len(gpus_rmvpe_list)
                for idx, n_g in enumerate(gpus_rmvpe_list):
                    cmd = (
                        f'python infer/modules/train/extract/extract_f0_rmvpe.py {leng} {idx} {n_g} "{log_dir}" {config.is_half}'
                    )
                    f0_cmds.append(cmd)
                if not run_process_parallel(f0_cmds, config):
                    logger.error("Parallel F0 extraction (rmvpe_gpu) failed.")
                    f0_success = False

        if not f0_success: return False
        logger.info("F0 extraction part completed.")

    logger.info("Starting feature extraction...")
    if not gpus_list or gpus_list == ['']:
        logger.error("No GPUs specified for feature extraction. Cannot proceed.")
        return False

    feature_cmds = []
    leng = len(gpus_list)
    for idx, n_g in enumerate(gpus_list):
        # В feature extraction передаем config.device напрямую, а не n_g
        # Скрипт extract_feature_print.py должен использовать первый аргумент как device
        cmd = (
            f'python infer/modules/train/extract_feature_print.py {config.device} {leng} {idx} "{log_dir}" {version} {config.is_half}'
        )
        # Убрали n_g из команды, предполагаем, что скрипт использует os.environ["CUDA_VISIBLE_DEVICES"] = idx (если GPU)
        # или просто первый аргумент как device. Нужно проверить extract_feature_print.py
        # Корректировка: Вернем n_g, так как скрипт его ожидает для установки CUDA_VISIBLE_DEVICES
        cmd = (
            f'python infer/modules/train/extract_feature_print.py {config.device} {leng} {idx} {n_g} "{log_dir}" {version} {config.is_half}'
        )
        feature_cmds.append(cmd)

    if not run_process_parallel(feature_cmds, config):
        logger.error("Feature extraction failed.")
        return False

    logger.info("Feature extraction completed.")
    return True