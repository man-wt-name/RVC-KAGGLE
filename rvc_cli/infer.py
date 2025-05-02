# --- rvc_cli/infer.py ---
import os
import logging
import traceback
import numpy as np
import torch
import soundfile as sf
from io import BytesIO
from infer.lib.audio import wav2 # Нужен для сохранения в mp3/m4a

logger = logging.getLogger(__name__)

def infer_one_cli(sid, input_audio, output_audio, transpose, f0_method,
                  index_path1, index_path2, index_rate, filter_radius,
                  resample_sr, rms_mix_rate, protect, f0_file, vc, config): # Передаем vc и config
    """CLI wrapper for single inference."""
    if not vc:
        logger.error("VC object not initialized. Cannot infer.")
        return False
    if not sid:
        logger.error("Speaker/model name (--sid) is required.")
        return False
    if not os.path.isfile(input_audio):
        logger.error(f"Input audio file not found: {input_audio}")
        return False
    if not output_audio:
        logger.error("Output audio path (--output_audio) is required.")
        return False

    final_index_path = index_path1 if index_path1 else index_path2
    if not final_index_path:
        logger.warning("No index file provided or selected. Retrieval quality may be affected.")
    elif not os.path.exists(final_index_path):
         logger.error(f"Index file not found: {final_index_path}")
         return False

    spk_id = 0
    final_protect = protect
    try:
        logger.info(f"Loading model: {sid}")
        get_vc_result = vc.get_vc(sid, protect, protect)
        spk_item_update = None
        if isinstance(get_vc_result, tuple) and len(get_vc_result) >= 5:
            spk_item_update = get_vc_result[0]
            protect1_update = get_vc_result[2]
            final_protect = protect1_update.get("value", protect) if isinstance(protect1_update, dict) else protect
        elif isinstance(get_vc_result, dict) and 'maximum' in get_vc_result:
            spk_item_update = get_vc_result
            final_protect = protect
        else:
             logger.warning("Could not determine speaker ID or protect value from get_vc. Using defaults.")

        if isinstance(spk_item_update, dict) and 'value' in spk_item_update:
             spk_id = spk_item_update.get("value", 0)
        elif isinstance(spk_item_update, dict) and 'maximum' in spk_item_update:
             spk_id = 0
             logger.warning("Only maximum speaker ID received, using default ID 0.")
        else:
             logger.warning("Could not determine speaker ID from get_vc. Using default ID 0.")

        logger.info(f"Using Speaker ID: {spk_id}, Protect: {final_protect}")

    except Exception as e:
        logger.error(f"Failed to load model or get speaker info for '{sid}': {e}")
        logger.error(traceback.format_exc())
        return False

    logger.info(f"Starting inference for {input_audio}...")
    success = False
    try:
        info_text, audio_out = vc.vc_single(
            spk_id, input_audio, transpose, f0_file, f0_method, final_index_path, "",
            index_rate, filter_radius, resample_sr, rms_mix_rate, final_protect
        )
        logger.info(f"Inference Info: {info_text}")
        if audio_out and isinstance(audio_out, tuple) and len(audio_out) == 2:
             sr_out, audio_data = audio_out
             try:
                 os.makedirs(os.path.dirname(output_audio), exist_ok=True)
                 sf.write(output_audio, audio_data, sr_out)
                 logger.info(f"Output audio saved to {output_audio} (Sample Rate: {sr_out})")
                 success = True
             except Exception as e:
                  logger.error(f"Failed to save output audio to {output_audio}: {e}")
        elif audio_out is None:
            logger.error(f"Inference failed. Check logs for details. Info: {info_text}")
        else:
            logger.warning(f"Inference completed but no valid audio output was generated. Info: {info_text}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
    return success

def infer_batch_cli(sid, input_paths, output_dir, transpose, f0_method,
                   index_path1, index_path2, index_rate, filter_radius,
                   resample_sr, rms_mix_rate, protect, format1, vc, config): # Добавлен format1
    """CLI wrapper for batch inference."""
    if not vc:
        logger.error("VC object not initialized. Cannot infer.")
        return False
    if not sid:
        logger.error("Speaker/model name (--sid) is required.")
        return False
    if not input_paths:
        logger.error("No input files or directory provided for batch inference.")
        return False

    final_index_path = index_path1 if index_path1 else index_path2
    if not final_index_path:
        logger.warning("No index file provided or selected. Retrieval quality may be affected.")
    elif not os.path.exists(final_index_path):
         logger.error(f"Index file not found: {final_index_path}")
         return False

    dir_input = None
    paths = []
    if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
        dir_input = input_paths[0]
        if not os.path.exists(dir_input):
            logger.error(f"Input directory not found: {dir_input}")
            return False
        try:
            # Собираем только поддерживаемые аудио файлы
            supported_exts = ('.wav', '.flac', '.mp3', '.m4a')
            paths = [os.path.join(dir_input, name) for name in os.listdir(dir_input) if name.lower().endswith(supported_exts)]
            if not paths:
                 logger.error(f"No supported audio files found in directory: {dir_input}")
                 return False
            logger.info(f"Processing directory: {dir_input}")
        except Exception as e:
            logger.error(f"Could not read directory {dir_input}: {e}")
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


    spk_id = 0
    final_protect = protect
    try:
        logger.info(f"Loading model: {sid}")
        get_vc_result = vc.get_vc(sid, protect, protect)
        spk_item_update = None
        if isinstance(get_vc_result, tuple) and len(get_vc_result) >= 5:
            spk_item_update = get_vc_result[0]
            protect1_update = get_vc_result[2]
            final_protect = protect1_update.get("value", protect) if isinstance(protect1_update, dict) else protect
        elif isinstance(get_vc_result, dict) and 'maximum' in get_vc_result:
            spk_item_update = get_vc_result
            final_protect = protect
        else:
             logger.warning("Could not determine speaker ID or protect value from get_vc. Using defaults.")

        if isinstance(spk_item_update, dict) and 'value' in spk_item_update:
             spk_id = spk_item_update.get("value", 0)
        elif isinstance(spk_item_update, dict) and 'maximum' in spk_item_update:
             spk_id = 0
             logger.warning("Only maximum speaker ID received, using default ID 0.")
        else:
             logger.warning("Could not determine speaker ID from get_vc. Using default ID 0.")

        logger.info(f"Using Speaker ID: {spk_id}, Protect: {final_protect}")
    except Exception as e:
        logger.error(f"Failed to load model or get speaker info for '{sid}': {e}")
        logger.error(traceback.format_exc())
        return False

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting batch inference...")
    all_successful = True
    try:
        # Передаем format1 в vc_multi
        info_generator = vc.vc_multi(
            spk_id, dir_input, output_dir, paths, transpose, f0_method, final_index_path, "",
            index_rate, filter_radius, resample_sr, rms_mix_rate, final_protect, format1
        )
        final_message = "Batch inference finished."
        for message in info_generator:
            logger.info(f"Batch Progress: {message}") # Логируем прогресс
            final_message = message
            if "->" in message and "Success" not in message:
                all_successful = False # Отмечаем если была ошибка

        logger.info(final_message)
        if all_successful:
            logger.info(f"Batch processing finished successfully. Check output directory: '{output_dir}'")
        else:
            logger.warning(f"Batch processing finished with errors. Check output directory: '{output_dir}' and logs.")

    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        logger.error(traceback.format_exc())
        all_successful = False
    return all_successful