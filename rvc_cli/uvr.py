# --- rvc_cli/uvr.py ---
import os
import logging
import traceback
import torch
# Импортируем функцию uvr из оригинального модуля
from infer.modules.uvr5.modules import uvr as uvr_core_func

logger = logging.getLogger(__name__)

# Переменные weight_uvr5_root и uvr5_names должны быть доступны
# Их можно либо передать, либо получить из config, либо определить здесь
# weight_uvr5_root = os.getenv("weight_uvr5_root", "assets/uvr5_weights")
# uvr5_names = [] # Заполнить как в run_rvc.py

def uvr_cli(model_name, input_paths, output_dir_vocals, output_dir_instrumental, agg, format, config, uvr5_names_list):
    """CLI wrapper for uvr."""
    weight_uvr5_root = os.getenv("weight_uvr5_root", "assets/uvr5_weights") # Получаем путь здесь

    if not model_name: # Проверка, если модель не задана (например, при вызове 'list')
        logger.error("UVR5 model name is required.")
        return False

    if model_name not in uvr5_names_list:
         model_file_path = os.path.join(weight_uvr5_root, f"{model_name}.pth")
         onnx_file_path = os.path.join(weight_uvr5_root, f"{model_name}.onnx")
         if not os.path.exists(model_file_path) and not os.path.exists(onnx_file_path):
             logger.error(f"Invalid UVR5 model name: {model_name}. Not found in known list or in {weight_uvr5_root}. Available: {uvr5_names_list}")
             return False
         else:
             logger.info(f"Using custom UVR5 model: {model_name}")

    if not input_paths:
         logger.error("No input files or directory provided for UVR5.")
         return False

    dir_wav_input = None
    paths = []
    if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
        dir_wav_input = input_paths[0]
        if not os.path.exists(dir_wav_input):
            logger.error(f"Input directory not found: {dir_wav_input}")
            return False
        try:
            supported_exts = ('.wav', '.flac', '.mp3', '.m4a')
            paths = [os.path.join(dir_wav_input, name) for name in os.listdir(dir_wav_input) if name.lower().endswith(supported_exts)]
            if not paths:
                logger.error(f"No supported audio files found in directory: {dir_wav_input}")
                return False
            logger.info(f"Processing directory: {dir_wav_input}")
        except Exception as e:
             logger.error(f"Could not read directory {dir_wav_input}: {e}")
             return False
    else:
        all_files_exist = True
        for p in input_paths:
            if os.path.isfile(p):
                paths.append(p)
            else:
                logger.error(f"Input file not found: {p}")
                all_files_exist = False
        if not all_files_exist:
            return False
        logger.info(f"Processing files: {paths}")

    os.makedirs(output_dir_vocals, exist_ok=True)
    os.makedirs(output_dir_instrumental, exist_ok=True)

    logger.info(f"Starting UVR5 processing with model {model_name}...")
    all_successful = True
    try:
        # Передаем config в uvr_core_func, если он там нужен
        # Предполагаем, что uvr_core_func использует config для device, is_half
        uvr_generator = uvr_core_func(model_name, dir_wav_input if dir_wav_input else "", output_dir_vocals, paths, output_dir_instrumental, agg, format)

        final_message = "UVR5 processing finished."
        for message in uvr_generator:
            logger.info(f"UVR Progress: {message}") # Логируем прогресс
            final_message = message
            if "->" in message and "Success" not in message:
                 all_successful = False # Отмечаем если была ошибка

        logger.info(final_message)
        if all_successful:
            logger.info(f"UVR processing finished successfully.")
        else:
             logger.warning(f"UVR processing finished with errors.")

    except Exception as e:
        logger.error(f"UVR processing failed: {e}")
        logger.error(traceback.format_exc())
        all_successful = False
    return all_successful