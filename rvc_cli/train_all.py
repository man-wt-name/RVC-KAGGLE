# --- rvc_cli/train_all.py ---
import logging
# Импортируем функции из других модулей rvc_cli
from rvc_cli.preprocess import preprocess_dataset_cli
from rvc_cli.extract import extract_f0_feature_cli
from rvc_cli.train import click_train_cli
from rvc_cli.index import train_index_cli

logger = logging.getLogger(__name__)

def train1key_cli(
    exp_dir_name, sr_str, if_f0, trainset_dir, spk_id, np7, f0method,
    save_epoch, total_epoch, batch_size, if_save_latest, pretrained_G,
    pretrained_D, gpus_str, if_cache_gpu, if_save_every_weights,
    version, gpus_rmvpe_str, config # Добавляем config
):
    """CLI version of train1key."""
    logger.info(f"--- Starting One-Key Training for experiment: {exp_dir_name} ---")

    logger.info(f"\n--- Step 1: Preprocessing Dataset ---")
    if not preprocess_dataset_cli(trainset_dir, exp_dir_name, sr_str, np7, config): # Передаем config
        logger.error("Preprocessing failed. Stopping training.")
        return

    logger.info(f"\n--- Step 2: Extracting Features (and F0 if enabled) ---")
    if not extract_f0_feature_cli(gpus_str, np7, f0method, if_f0, exp_dir_name, version, gpus_rmvpe_str, config): # Передаем config
         logger.error("Feature/F0 extraction failed. Stopping training.")
         return

    logger.info(f"\n--- Step 3a: Training Model ---")
    if not click_train_cli(
        exp_dir_name, sr_str, if_f0, spk_id, save_epoch, total_epoch,
        batch_size, if_save_latest, pretrained_G, pretrained_D,
        gpus_str, if_cache_gpu, if_save_every_weights, version, config # Передаем config
    ):
        logger.error("Model training failed. Stopping.")
        return

    logger.info(f"\n--- Step 3b: Training Index ---")
    if not train_index_cli(exp_dir_name, version, config): # Передаем config
        logger.error("Index training failed.")
    else:
        logger.info("Index training successful.")

    logger.info(f"\n--- One-Key Training for '{exp_dir_name}' Completed! ---")