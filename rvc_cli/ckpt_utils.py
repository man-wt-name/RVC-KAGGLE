# --- rvc_cli/ckpt_utils.py ---
import os
import logging
import traceback
import torch
from infer.lib.train.process_ckpt import ( # Импортируем нужные функции
    change_info,
    extract_small_model,
    merge,
    show_info,
)

logger = logging.getLogger(__name__)
# weight_root должен быть доступен
# weight_root = os.getenv("weight_root", "assets/weights")

def merge_cli(ckpt_a_path, ckpt_b_path, alpha, sr_str, if_f0, info_str, save_name, version, config):
    """CLI wrapper for merge."""
    weight_root = os.getenv("weight_root", "assets/weights")
    if not save_name:
        logger.error("Save name for merged model is required (--save_name).")
        return False
    save_path = os.path.join(weight_root, f"{save_name}.pth")
    logger.info(f"Merging {ckpt_a_path} and {ckpt_b_path} with alpha={alpha} into {save_path}")
    success = False
    try:
        result = merge(ckpt_a_path, ckpt_b_path, alpha, sr_str, if_f0, info_str, save_name, version)
        logger.info(result)
        if "Success" in result:
             success = True
    except Exception as e:
        logger.error(f"Merging failed: {e}")
        logger.error(traceback.format_exc())
    return success

def change_info_cli(ckpt_path, info_str, save_name, config):
    """CLI wrapper for change_info."""
    weight_root = os.getenv("weight_root", "assets/weights")
    logger.info(f"Changing info for {ckpt_path}")
    final_save_name = save_name if save_name else os.path.basename(ckpt_path)
    save_path = os.path.join(weight_root, final_save_name) # Путь теперь полный
    if not save_name: # Если имя не задано, перезаписываем исходный
        save_path = ckpt_path
        final_save_name = os.path.basename(ckpt_path) # Используем только имя файла для лога
        logger.warning(f"No save name provided, overwriting original file: {ckpt_path}")

    logger.info(f"New info: '{info_str}'. Saving as '{final_save_name}' in '{weight_root}'")
    success = False
    try:
        # Передаем только имя файла для сохранения в `weights`
        result = change_info(ckpt_path, info_str, final_save_name)
        logger.info(result)
        if "Success" in result:
            success = True
    except Exception as e:
        logger.error(f"Changing info failed: {e}")
        logger.error(traceback.format_exc())
    return success

def show_info_cli(ckpt_path, config):
    """CLI wrapper for show_info."""
    logger.info(f"Showing info for {ckpt_path}")
    success = False
    try:
        if not os.path.exists(ckpt_path):
             logger.error(f"Checkpoint file not found: {ckpt_path}")
             return False
        result = show_info(ckpt_path)
        print(f"--- Model Info: {ckpt_path} ---\n{result}\n-----------------------------")
        success = True
    except Exception as e:
        logger.error(f"Showing info failed: {e}")
        logger.error(traceback.format_exc())
    return success

def extract_small_model_cli(ckpt_path, save_name, sr_str, if_f0_str, info_str, version, config):
    """CLI wrapper for extract_small_model."""
    weight_root = os.getenv("weight_root", "assets/weights")
    if not save_name:
        logger.error("Save name for extracted model is required (--save_name).")
        return False
    if if_f0_str not in ["0", "1"]:
         logger.error("Invalid value for --if_f0. Must be '0' or '1'.")
         return False
    # Полный путь не нужен для extract_small_model, она сохраняет в assets/weights
    # save_path = os.path.join(weight_root, f"{save_name}.pth")
    logger.info(f"Extracting small model from {ckpt_path} to 'assets/weights/{save_name}.pth'")
    success = False
    try:
        result = extract_small_model(ckpt_path, save_name, sr_str, if_f0_str, info_str, version)
        logger.info(result)
        if "Success" in result:
            success = True
    except Exception as e:
        logger.error(f"Extracting small model failed: {e}")
        logger.error(traceback.format_exc())
    return success