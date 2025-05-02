# --- rvc_cli/train.py ---
import os
import logging
import json
import pathlib
from random import shuffle
from rvc_cli.helpers import run_process, get_pretrained_models # Импортируем helpers

logger = logging.getLogger(__name__)
now_dir = os.getcwd()

# Словарь sr_dict можно импортировать или определить здесь
sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def click_train_cli(
    exp_dir_name, sr_str, if_f0, spk_id, save_epoch, total_epoch, batch_size,
    if_save_latest, pretrained_G, pretrained_D, gpus_str, if_cache_gpu,
    if_save_every_weights, version, config
):
    """CLI version of click_train."""
    if sr_str not in sr_dict:
        logger.error(f"Invalid sample rate: {sr_str}. Choose from {list(sr_dict.keys())}")
        return False
    sr = sr_str # Keep string version

    exp_dir = os.path.join(now_dir, "logs", exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, f"3_feature{'256' if version == 'v1' else '768'}")

    if not os.path.exists(gt_wavs_dir) or not os.path.exists(feature_dir):
         logger.error(f"Required directories '{gt_wavs_dir}' or '{feature_dir}' not found. Run preprocessing and feature extraction first.")
         return False

    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    if if_f0 and (not os.path.exists(f0_dir) or not os.path.exists(f0nsf_dir)):
        logger.error(f"F0 is enabled, but required directories '{f0_dir}' or '{f0nsf_dir}' not found. Run F0 extraction first.")
        return False

    logger.info("Generating filelist...")
    try:
        gt_names = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
        feature_names = set(name.split(".")[0] for name in os.listdir(feature_dir))
        common_names = gt_names & feature_names
        if if_f0:
            f0_names = set(name.split(".")[0] for name in os.listdir(f0_dir))
            f0nsf_names = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
            common_names &= f0_names & f0nsf_names
    except Exception as e:
        logger.error(f"Error accessing processed directories: {e}")
        return False

    opt = []
    for name in common_names:
        paths = [
            os.path.join(gt_wavs_dir, f"{name}.wav"),
            os.path.join(feature_dir, f"{name}.npy"),
        ]
        if if_f0:
            paths.extend([
                os.path.join(f0_dir, f"{name}.wav.npy"),
                os.path.join(f0nsf_dir, f"{name}.wav.npy"),
            ])
        paths.append(str(spk_id))
        opt.append("|".join(p.replace("\\", "/") for p in paths))

    if not opt:
         logger.error("No common files found after preprocessing/extraction. Cannot generate filelist.")
         return False

    fea_dim = 256 if version == "v1" else 768
    mute_base = os.path.join(now_dir, "logs", "mute")
    mute_gt = os.path.join(mute_base, "0_gt_wavs", f"mute{sr}.wav").replace("\\", "/")
    mute_feat = os.path.join(mute_base, f"3_feature{fea_dim}", "mute.npy").replace("\\", "/")

    if not os.path.exists(mute_gt) or not os.path.exists(mute_feat):
        logger.warning(f"Mute files not found in {mute_base}. Skipping adding mute samples.")
    else:
        if if_f0:
            mute_f0 = os.path.join(mute_base, "2a_f0", "mute.wav.npy").replace("\\", "/")
            mute_f0nsf = os.path.join(mute_base, "2b-f0nsf", "mute.wav.npy").replace("\\", "/")
            if not os.path.exists(mute_f0) or not os.path.exists(mute_f0nsf):
                logger.warning(f"Mute F0 files not found in {mute_base}. Skipping adding mute samples.")
            else:
                for _ in range(2):
                    opt.append(f"{mute_gt}|{mute_feat}|{mute_f0}|{mute_f0nsf}|{spk_id}")
        else:
            for _ in range(2):
                 opt.append(f"{mute_gt}|{mute_feat}|{spk_id}")

    shuffle(opt)
    filelist_path = os.path.join(exp_dir, "filelist.txt")
    try:
        with open(filelist_path, "w", encoding="utf-8") as f:
            f.write("\n".join(opt))
        logger.info(f"Filelist generated successfully at {filelist_path}")
    except Exception as e:
        logger.error(f"Failed to write filelist: {e}")
        return False

    if version == "v1" or sr == "40k":
        config_path_json = f"v1/{sr}.json"
    else:
        config_path_json = f"v2/{sr}.json"

    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        logger.info(f"Generating config file at {config_save_path}")
        if config_path_json not in config.json_config:
             logger.error(f"Base config '{config_path_json}' not found in internal config. Cannot generate config.json.")
             return False

        try:
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(
                    config.json_config[config_path_json], f,
                    ensure_ascii=False, indent=4, sort_keys=True
                )
        except KeyError:
            logger.error(f"Could not find base config '{config_path_json}' in internal config. Cannot proceed.")
            return False
        except Exception as e:
             logger.error(f"Failed to write config file: {e}")
             return False

    # --- Resolve pretrained paths ---
    path_str = "" if version == "v1" else "_v2"
    f0_str_pre = "f0" if if_f0 else ""
    sr_for_pretrain = sr_str

    auto_pretrained_G, auto_pretrained_D = get_pretrained_models(path_str, f0_str_pre, sr_for_pretrain)

    final_pretrained_G = pretrained_G if pretrained_G else auto_pretrained_G
    final_pretrained_D = pretrained_D if pretrained_D else auto_pretrained_D

    if final_pretrained_G: logger.info(f"Using Pretrained G: {final_pretrained_G}")
    else: logger.warning("No Pretrained G specified or found.")
    if final_pretrained_D: logger.info(f"Using Pretrained D: {final_pretrained_D}")
    else: logger.warning("No Pretrained D specified or found.")
    # --- Construct train command ---
    cmd = (
        f'python infer/modules/train/train.py '
        f'-e "{exp_dir_name}" -sr {sr} -f0 {1 if if_f0 else 0} -bs {batch_size} '
        f'{"-g " + gpus_str if gpus_str else ""} '
        f'-te {total_epoch} -se {save_epoch} '
        f'{"-pg " + final_pretrained_G if final_pretrained_G else ""} '
        f'{"-pd " + final_pretrained_D if final_pretrained_D else ""} '
        f'-l {1 if if_save_latest else 0} '
        f'-c {1 if if_cache_gpu else 0} '
        f'-sw {1 if if_save_every_weights else 0} '
        f'-v {version}'
    )

    # Run training process using helper
    if run_process(cmd, config) != 0:
        logger.error(f"Training process failed for experiment '{exp_dir_name}'.")
        return False
    logger.info(f"Training process for experiment '{exp_dir_name}' finished. Check logs in '{exp_dir}'.")
    return True