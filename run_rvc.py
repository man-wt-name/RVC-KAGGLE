# -*- coding: utf-8 -*-
import os
import sys
import argparse # Добавлено
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

# --- Оставлены необходимые импорты ---
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from i18n.i18n import I18nAuto # Оставлено для возможных логов
from configs.config import Config # Импорт Config остается
from sklearn.cluster import MiniBatchKMeans
import torch, platform
import numpy as np
# import gradio as gr # Удалено
import faiss
import fairseq
import pathlib
import json
from time import sleep
from subprocess import Popen, PIPE, STDOUT # Добавлен PIPE, STDOUT
from random import shuffle
import warnings
import traceback
import threading # Оставлено для мониторинга Popen
import shutil
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
# Настройка базового логгера для вывода в консоль
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514) # Оставлено для воспроизводимости

# --- ДОБАВЛЕНО: Определение sr_dict в глобальной области видимости ---
sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}
# --------------------------------------------------------------------

# Инициализация Config ПОСЛЕ разбора основных аргументов в main()
config = None # Placeholder
vc = None # Placeholder

# ########## Инициализация Config и VC перенесена в main() ############


if config and config.dml == True: # Проверка на None
    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res
    if fairseq and hasattr(fairseq, "modules") and hasattr(fairseq.modules, "grad_multiply"):
         fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
    else:
         logger.warning("Could not patch fairseq for DML.")


i18n = I18nAuto() # Оставлено для возможных логов
# logger.info(f"Internationalization method: {i18n.language}") # Пример использования

# Определение GPU (логика оставлена)
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# Эта проверка выполняется до инициализации Config, но config.device там тоже определяется
# Возможно, стоит перенести определение device из Config сюда или инициализировать Config раньше,
# но тогда вернемся к проблеме парсинга. Оставим пока так.
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        # Проверка на поддерживаемые GPU (оставлена, но все еще хрупкая)
        if any(
            value in gpu_name.upper()
            for value in [
                "10", "16", "20", "30", "40", "A2", "A3", "A4", "P4",
                "A50", "500", "A60", "70", "80", "90", "M4", "T4",
                "TITAN", "4060", "L", "6000",
            ]
        ):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024 / 1024 / 1024 + 0.4
                )
            )

gpu_info="" # Инициализируем переменную
default_batch_size = 1 # Значение по умолчанию
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2 if mem else 1 # Добавлена проверка на пустой mem
else:
    gpu_info = "No suitable NVIDIA GPU found for training." # Заменено i18n
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos]) if gpu_infos else "" # Добавлена проверка


# Загрузка путей и имен файлов (логика оставлена)
# Используем значения по умолчанию, если переменные окружения не установлены
weight_root = os.getenv("weight_root", "assets/weights")
weight_uvr5_root = os.getenv("weight_uvr5_root", "assets/uvr5_weights")
index_root = os.getenv("index_root", "logs")
outside_index_root = os.getenv("outside_index_root", ".") # Путь по умолчанию для ссылок

names = []
if weight_root and os.path.exists(weight_root):
    try:
        for name in os.listdir(weight_root):
            if name.endswith(".pth"):
                names.append(name)
    except Exception as e:
        logger.warning(f"Could not list weights in {weight_root}: {e}")
else:
    logger.warning(f"Weight root directory '{weight_root}' not found or not set.")

index_paths = []
def lookup_indices(root_dir):
    if not root_dir or not os.path.exists(root_dir):
        logger.warning(f"Index directory '{root_dir}' not found or not set.")
        return
    try:
        for root, dirs, files in os.walk(root_dir, topdown=False):
            for name in files:
                if name.endswith(".index") and "trained" not in name:
                    index_paths.append(os.path.join(root, name)) # Исправлено на os.path.join
    except Exception as e:
        logger.warning(f"Could not list indices in {root_dir}: {e}")

lookup_indices(index_root)
# lookup_indices(outside_index_root) # Не ищем во внешней директории по умолчанию

uvr5_names = []
if weight_uvr5_root and os.path.exists(weight_uvr5_root):
    try:
        for name in os.listdir(weight_uvr5_root):
            if name.endswith(".pth") or "onnx" in name:
                uvr5_names.append(name.replace(".pth", ""))
    except Exception as e:
        logger.warning(f"Could not list UVR5 weights in {weight_uvr5_root}: {e}")
else:
    logger.warning(f"UVR5 weight root directory '{weight_uvr5_root}' not found or not set.")


# --- Функции if_done, if_done_multi оставлены для Popen ---
def if_done(done, p):
    while True:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True

def if_done_multi(done, ps):
    while True:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

# --- Модифицированные функции для работы без Gradio ---

def run_process(cmd):
    """Helper to run subprocess and stream output."""
    logger.info(f"Executing: {cmd}")
    try:
        # Используем PIPE для перехвата вывода
        # Добавляем env=os.environ чтобы передать текущие переменные окружения
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, universal_newlines=True, cwd=now_dir, env=os.environ)
        # Читаем вывод построчно
        if p.stdout:
            for line in iter(p.stdout.readline, ''):
                print(line.strip()) # Выводим в консоль
        p.wait()
        if p.returncode != 0:
            logger.error(f"Command '{cmd}' failed with return code {p.returncode}")
        else:
             logger.info(f"Command '{cmd}' finished successfully.")
        return p.returncode
    except Exception as e:
        logger.error(f"Failed to execute command '{cmd}': {e}")
        logger.error(traceback.format_exc())
        return -1


def run_process_parallel(cmds):
    """Helper to run multiple subprocesses in parallel and stream output."""
    processes = []
    outputs = {} # Словарь для хранения вывода каждого процесса

    # Функция для чтения вывода процесса
    def reader_thread(p, pid, output_list):
        if p.stdout:
            for line in iter(p.stdout.readline, ''):
                output_list.append(line)
                print(f"[PID {pid}]: {line.strip()}") # Печатаем с PID
        rc = p.wait()
        logger.info(f"Process PID {pid} finished with return code {rc}.")

    threads = []
    logger.info("Starting parallel processes...")
    for i, cmd in enumerate(cmds):
        logger.info(f"Executing [{i+1}/{len(cmds)}]: {cmd}")
        try:
             # Добавляем env=os.environ
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, universal_newlines=True, cwd=now_dir, env=os.environ)
            pid = p.pid
            outputs[pid] = []
            thread = threading.Thread(target=reader_thread, args=(p, pid, outputs[pid]))
            thread.start()
            threads.append(thread)
            processes.append(p) # Сохраняем объект процесса
        except Exception as e:
            logger.error(f"Failed to start command '{cmd}': {e}")

    # Ждем завершения всех потоков
    for thread in threads:
        thread.join()

    # Проверяем коды возврата
    all_successful = True
    failed_cmds = []
    for i, p in enumerate(processes):
        if p.returncode != 0:
            logger.error(f"Command '{cmds[i]}' (PID {p.pid}) failed with return code {p.returncode}")
            failed_cmds.append(cmds[i])
            all_successful = False
        # Можно собрать весь вывод: full_output = "".join(outputs[p.pid])

    if all_successful:
         logger.info("All parallel processes finished successfully.")
    else:
         logger.warning(f"One or more parallel processes failed: {failed_cmds}")

    return all_successful


def preprocess_dataset_cli(trainset_dir, exp_dir_name, sr_str, n_p):
    """CLI version of preprocess_dataset."""
    global config # Используем глобальный config
    if sr_str not in sr_dict:
        logger.error(f"Invalid sample rate: {sr_str}. Choose from {list(sr_dict.keys())}")
        return False
    sr = sr_dict[sr_str]
    log_dir = os.path.join(now_dir, "logs", exp_dir_name)
    os.makedirs(log_dir, exist_ok=True)

    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        log_dir,
        config.noparallel,
        config.preprocess_per,
    )
    return run_process(cmd) == 0 # Возвращаем True при успехе, False при ошибке


def extract_f0_feature_cli(gpus_str, n_p, f0method, if_f0, exp_dir_name, version, gpus_rmvpe_str):
    """CLI version of extract_f0_feature."""
    global config # Используем глобальный config
    gpus_list = gpus_str.split("-") if gpus_str else []
    log_dir = os.path.join(now_dir, "logs", exp_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    # f0_log_path = os.path.join(log_dir, "extract_f0_feature.log") # Лог теперь пишется самим скриптом в stdout/stderr

    # F0 Extraction part
    f0_success = True
    if if_f0:
        logger.info("Starting F0 extraction...")
        if f0method != "rmvpe_gpu":
            # Другие методы F0 (pm, harvest, dio, rmvpe cpu)
             cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s" %s %s'
                % (
                    config.python_cmd,
                    log_dir,
                    n_p,
                    f0method,
                )
            )
             if run_process(cmd) != 0:
                 logger.error(f"F0 extraction ({f0method}) failed.")
                 f0_success = False # Отмечаем неудачу

        else:
            # RMVPE GPU parallel execution
            gpus_rmvpe_list = gpus_rmvpe_str.split("-") if gpus_rmvpe_str else []
            if not gpus_rmvpe_list or gpus_rmvpe_list == ['']: # Проверка на пустой список или список с пустой строкой
                 logger.warning("No GPUs specified for rmvpe_gpu. F0 extraction might be slow or fail.")
                 # Здесь можно добавить fallback на CPU rmvpe или другой метод, если нужно
            elif gpus_rmvpe_list == ["-"]:
                 logger.warning("GPU usage explicitly disabled for rmvpe_gpu ('-'). Skipping GPU F0 extraction.")
            # elif config.dml == True: # DirectML version - пока не поддерживается параллельно
            #     cmd = (
            #         config.python_cmd
            #         + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s" '
            #         % (log_dir,)
            #     )
            #     if run_process(cmd) != 0:
            #         logger.error("F0 extraction (rmvpe_dml) failed.")
            #         f0_success = False
            else: # Запуск rmvpe_gpu параллельно
                f0_cmds = []
                leng = len(gpus_rmvpe_list)
                for idx, n_g in enumerate(gpus_rmvpe_list):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s" %s '
                        % (
                            config.python_cmd, leng, idx, n_g, log_dir, config.is_half,
                        )
                    )
                    f0_cmds.append(cmd)
                if not run_process_parallel(f0_cmds):
                    logger.error("Parallel F0 extraction (rmvpe_gpu) failed.")
                    f0_success = False # Отмечаем неудачу

        if not f0_success: return False # Прерываем, если F0 не удался
        logger.info("F0 extraction part completed.")

    # Feature Extraction part
    logger.info("Starting feature extraction...")
    if not gpus_list or gpus_list == ['']:
        logger.error("No GPUs specified for feature extraction. Cannot proceed.")
        return False

    feature_cmds = []
    leng = len(gpus_list)
    for idx, n_g in enumerate(gpus_list):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s" %s %s'
            % (
                config.python_cmd,
                config.device, leng, idx, n_g, log_dir, version, config.is_half,
            )
        )
        feature_cmds.append(cmd)

    if not run_process_parallel(feature_cmds):
        logger.error("Feature extraction failed.")
        return False

    logger.info("Feature extraction completed.")
    return True # Возвращаем успех


def get_pretrained_models(path_str, f0_str, sr2):
    """Remains the same, just logging."""
    g_path = os.path.join("assets", f"pretrained{path_str}", f"{f0_str}G{sr2}.pth")
    d_path = os.path.join("assets", f"pretrained{path_str}", f"{f0_str}D{sr2}.pth")

    if not os.access(g_path, os.F_OK):
        logger.warning(f"{g_path} not found, will not use pretrained Generator.")
        g_path = ""
    if not os.access(d_path, os.F_OK):
        logger.warning(f"{d_path} not found, will not use pretrained Discriminator.")
        d_path = ""
    return g_path, d_path

# --- change_sr2, change_version19, change_f0 не нужны для CLI ---

def click_train_cli(
    exp_dir_name, sr_str, if_f0, spk_id, save_epoch, total_epoch, batch_size,
    if_save_latest, pretrained_G, pretrained_D, gpus_str, if_cache_gpu,
    if_save_every_weights, version
):
    """CLI version of click_train."""
    global config # Используем глобальный config
    if sr_str not in sr_dict:
        logger.error(f"Invalid sample rate: {sr_str}. Choose from {list(sr_dict.keys())}")
        return False
    sr = sr_str # Keep string version for config path

    exp_dir = os.path.join(now_dir, "logs", exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, f"3_feature{'256' if version == 'v1' else '768'}")

    if not os.path.exists(gt_wavs_dir) or not os.path.exists(feature_dir):
         logger.error(f"Required directories '{gt_wavs_dir}' or '{feature_dir}' not found. Run preprocessing and feature extraction first.")
         return False

    # Check F0 dirs if needed
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    if if_f0 and (not os.path.exists(f0_dir) or not os.path.exists(f0nsf_dir)):
        logger.error(f"F0 is enabled, but required directories '{f0_dir}' or '{f0nsf_dir}' not found. Run F0 extraction first.")
        return False

    # Generate filelist
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
        # Use forward slashes for compatibility within the training script if needed
        opt.append("|".join(p.replace("\\", "/") for p in paths))

    if not opt:
         logger.error("No common files found after preprocessing/extraction. Cannot generate filelist.")
         return False

    # Add mute samples
    fea_dim = 256 if version == "v1" else 768
    mute_base = os.path.join(now_dir, "logs", "mute") # Базовый путь к mute
    mute_gt = os.path.join(mute_base, "0_gt_wavs", f"mute{sr}.wav").replace("\\", "/")
    mute_feat = os.path.join(mute_base, f"3_feature{fea_dim}", "mute.npy").replace("\\", "/")

    # Проверяем наличие mute файлов
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
        with open(filelist_path, "w", encoding="utf-8") as f: # Добавлено utf-8
            f.write("\n".join(opt))
        logger.info(f"Filelist generated successfully at {filelist_path}")
    except Exception as e:
        logger.error(f"Failed to write filelist: {e}")
        return False


    # Generate config if it doesn't exist
    if version == "v1" or sr == "40k": # Assuming sr is still like "40k"
        config_path_json = f"v1/{sr}.json"
    else:
        config_path_json = f"v2/{sr}.json"

    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        logger.info(f"Generating config file at {config_save_path}")
        # Проверяем наличие базового конфига в config.json_config
        if config_path_json not in config.json_config:
             logger.error(f"Base config '{config_path_json}' not found in internal config. Cannot generate config.json.")
             return False # Прерываем, если базовый конфиг не найден

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

    # Construct train command
    cmd = (
        f'"{config.python_cmd}" infer/modules/train/train.py '
        f'-e "{exp_dir_name}" -sr {sr} -f0 {1 if if_f0 else 0} -bs {batch_size} '
        f'{"-g " + gpus_str if gpus_str else ""} ' # Only add -g if gpus are specified
        f'-te {total_epoch} -se {save_epoch} '
        f'{"-pg " + pretrained_G if pretrained_G else ""} '
        f'{"-pd " + pretrained_D if pretrained_D else ""} '
        f'-l {1 if if_save_latest else 0} '
        f'-c {1 if if_cache_gpu else 0} '
        f'-sw {1 if if_save_every_weights else 0} '
        f'-v {version}'
    )

    # Run training process
    if run_process(cmd) != 0:
        logger.error(f"Training process failed for experiment '{exp_dir_name}'.")
        return False
    logger.info(f"Training process for experiment '{exp_dir_name}' finished. Check logs in '{exp_dir}'.")
    return True # Возвращаем успех


def train_index_cli(exp_dir_name, version):
    """CLI version of train_index."""
    global config # Используем глобальный config
    exp_dir = os.path.join("logs", exp_dir_name) # Relative path as used in original
    feature_dir = os.path.join(exp_dir, f"3_feature{'256' if version == 'v1' else '768'}")

    if not os.path.exists(feature_dir):
        logger.error(f"Feature directory '{feature_dir}' not found. Please run feature extraction first.")
        return False
    try:
        listdir_res = list(os.listdir(feature_dir))
        if not listdir_res:
            logger.error(f"Feature directory '{feature_dir}' is empty.")
            return False
    except Exception as e:
        logger.error(f"Cannot read feature directory '{feature_dir}': {e}")
        return False

    logger.info("Starting index training...")
    npys = []
    # Исправлена ошибка - неправильный путь к файлу внутри цикла
    for name in sorted(listdir_res):
        if not name.endswith(".npy"): continue
        file_path = os.path.join(feature_dir, name)
        try:
            phone = np.load(file_path)
            npys.append(phone)
        except Exception as e:
            logger.warning(f"Could not load feature file {file_path}: {e}")


    if not npys:
        logger.error("No valid .npy files found in feature directory.")
        return False

    try:
        big_npy = np.concatenate(npys, 0)
    except ValueError as e:
        logger.error(f"Error concatenating numpy arrays: {e}. Check feature file integrity.")
        return False


    if big_npy.shape[0] == 0:
        logger.error("Concatenated features are empty.")
        return False

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    index_dim = 256 if version == "v1" else 768

    if big_npy.shape[1] != index_dim:
        logger.error(f"Feature dimension mismatch: expected {index_dim}, got {big_npy.shape[1]}")
        return False

    # Убеждаемся, что тип данных float32 для FAISS и KMeans
    if big_npy.dtype != np.float32:
        logger.warning(f"Feature dtype is {big_npy.dtype}, converting to float32.")
        big_npy = big_npy.astype(np.float32)


    if big_npy.shape[0] > 2e5:
        logger.info(f"Dataset size ({big_npy.shape[0]}) > 200k, applying MiniBatchKMeans...")
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=10000, verbose=True, batch_size=256 * config.n_cpu,
                compute_labels=False, init="random", n_init='auto' # Added n_init
            )
            big_npy = kmeans.fit(big_npy).cluster_centers_
             # Убеждаемся, что после kmeans тип данных остался float32
            if big_npy.dtype != np.float32:
                big_npy = big_npy.astype(np.float32)
            logger.info(f"KMeans finished, reduced to {big_npy.shape[0]} centers.")
        except Exception as e:
            logger.error(f"KMeans failed: {e}\n{traceback.format_exc()}")
            logger.warning("Continuing index training with original full features.")

    index_save_base = os.path.join(exp_dir, f"total_fea_{exp_dir_name}_{version}")
    try:
        np.save(f"{index_save_base}.npy", big_npy)
        logger.info(f"Saved features for indexing to {index_save_base}.npy")
    except Exception as e:
        logger.error(f"Failed to save features for indexing: {e}")
        return False


    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    if n_ivf < 1:
        logger.warning(f"Calculated n_ivf ({n_ivf}) is too small, setting to 1.")
        n_ivf = 1
    logger.info(f"Feature shape: {big_npy.shape}, calculated n_ivf: {n_ivf}")

    index = None
    index_ivf = None # Инициализация
    try:
        index = faiss.index_factory(index_dim, f"IVF{n_ivf},Flat")
        logger.info("Training FAISS index structure...")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = max(1, int(np.power(n_ivf, 0.3))) # Альтернативная эвристика для nprobe
        # index_ivf.nprobe = max(1, n_ivf // 10) # Set nprobe dynamically, e.g., 10% of n_ivf or minimum 1
        logger.info(f"Set nprobe to {index_ivf.nprobe}")
        index.train(big_npy)
        trained_index_path = os.path.join(exp_dir, f"trained_IVF{n_ivf}_Flat_nprobe{index_ivf.nprobe}_{exp_dir_name}_{version}.index")
        faiss.write_index(index, trained_index_path)
        logger.info(f"Index structure trained and saved to {trained_index_path}")

        logger.info("Adding features to index...")
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])
            if (i // batch_size_add) % 10 == 0:
                 logger.info(f"Added {min(i + batch_size_add, big_npy.shape[0])}/{big_npy.shape[0]} vectors")

        final_index_path = os.path.join(exp_dir, f"added_IVF{n_ivf}_Flat_nprobe{index_ivf.nprobe}_{exp_dir_name}_{version}.index")
        faiss.write_index(index, final_index_path)
        logger.info(f"Index populated and saved to {final_index_path}")

        # Create link in outside_index_root
        if outside_index_root and os.path.exists(outside_index_root):
            # link_func = os.link if platform.system() == "Windows" else os.symlink # os.link может требовать прав администратора
            link_func = shutil.copyfile # Используем копирование вместо ссылки для надежности
            link_name = os.path.join(outside_index_root, os.path.basename(final_index_path))
            try:
                if os.path.exists(link_name):
                     logger.warning(f"Link target '{link_name}' already exists, overwriting.")
                     os.remove(link_name)
                link_func(final_index_path, link_name)
                logger.info(f"Successfully copied index to '{link_name}'")
            except Exception as e:
                logger.error(f"Failed to copy index to '{link_name}': {e}")
        else:
            logger.warning(f"Cannot copy index: outside_index_root '{outside_index_root}' not found or not set.")

    except Exception as e:
        logger.error(f"FAISS index training/adding failed: {e}")
        logger.error(traceback.format_exc())
        return False # Возвращаем ошибку
    finally:
        # Clean up index object
        if index: del index
        if index_ivf: del index_ivf # Удаляем index_ivf
        if 'kmeans' in locals() and 'kmeans' in vars(): del kmeans # Проверяем перед удалением
        import gc
        gc.collect() # Try to free memory
    return True # Возвращаем успех


def train1key_cli(
    exp_dir_name, sr_str, if_f0, trainset_dir, spk_id, np7, f0method,
    save_epoch, total_epoch, batch_size, if_save_latest, pretrained_G,
    pretrained_D, gpus16_str, if_cache_gpu, if_save_every_weights,
    version, gpus_rmvpe_str
):
    """CLI version of train1key."""
    logger.info(f"--- Starting One-Key Training for experiment: {exp_dir_name} ---")

    logger.info(f"\n--- Step 1: Preprocessing Dataset ---")
    if not preprocess_dataset_cli(trainset_dir, exp_dir_name, sr_str, np7):
        logger.error("Preprocessing failed. Stopping training.")
        return

    logger.info(f"\n--- Step 2: Extracting Features (and F0 if enabled) ---")
    if not extract_f0_feature_cli(gpus16_str, np7, f0method, if_f0, exp_dir_name, version, gpus_rmvpe_str):
         logger.error("Feature/F0 extraction failed. Stopping training.")
         return


    logger.info(f"\n--- Step 3a: Training Model ---")
    # Resolve pretrained paths dynamically
    path_str = "" if version == "v1" else "_v2"
    f0_str_pre = "f0" if if_f0 else ""
    sr_for_pretrain = sr_str
    # if version == "v1" and sr_str == "32k": # Убрано, так как нет претрейна для 32k v1
    #      sr_for_pretrain = "40k"
    #      logger.info("Using 40k pretrained models for v1 32k training.")

    auto_pretrained_G, auto_pretrained_D = get_pretrained_models(path_str, f0_str_pre, sr_for_pretrain)

    # Use provided paths if given, otherwise use auto-detected ones
    final_pretrained_G = pretrained_G if pretrained_G else auto_pretrained_G
    final_pretrained_D = pretrained_D if pretrained_D else auto_pretrained_D

    if final_pretrained_G: logger.info(f"Using Pretrained G: {final_pretrained_G}")
    else: logger.warning("No Pretrained G specified or found.")
    if final_pretrained_D: logger.info(f"Using Pretrained D: {final_pretrained_D}")
    else: logger.warning("No Pretrained D specified or found.")


    if not click_train_cli(
        exp_dir_name, sr_str, if_f0, spk_id, save_epoch, total_epoch,
        batch_size, if_save_latest, final_pretrained_G, final_pretrained_D,
        gpus16_str, if_cache_gpu, if_save_every_weights, version
    ):
        logger.error("Model training failed. Stopping.")
        return


    logger.info(f"\n--- Step 3b: Training Index ---")
    if not train_index_cli(exp_dir_name, version):
        logger.error("Index training failed.")
        # Не прерываем, модель обучена, но индекса нет
    else:
        logger.info("Index training successful.")


    logger.info(f"\n--- One-Key Training for '{exp_dir_name}' Completed! ---")


# --- Ckpt functions (merge, change_info, show_info, extract_small_model) can be used directly ---
# --- Need CLI wrappers for them ---

def merge_cli(ckpt_a_path, ckpt_b_path, alpha, sr_str, if_f0, info_str, save_name, version):
    """CLI wrapper for merge."""
    global config # Используем глобальный config
    if not save_name:
        logger.error("Save name for merged model is required (--save_name).")
        return
    save_path = os.path.join(weight_root, f"{save_name}.pth") # Полный путь
    logger.info(f"Merging {ckpt_a_path} and {ckpt_b_path} with alpha={alpha} into {save_path}")
    try:
        # Передаем sr_str как строку, if_f0 как bool
        result = merge(ckpt_a_path, ckpt_b_path, alpha, sr_str, if_f0, info_str, save_name, version)
        logger.info(result) # merge function returns status string
    except Exception as e:
        logger.error(f"Merging failed: {e}")
        logger.error(traceback.format_exc())

def change_info_cli(ckpt_path, info_str, save_name):
    """CLI wrapper for change_info."""
    global config # Используем глобальный config
    logger.info(f"Changing info for {ckpt_path}")
    final_save_name = save_name if save_name else os.path.basename(ckpt_path) # Имя файла для сохранения
    save_path = os.path.join(weight_root, final_save_name) # Полный путь
    logger.info(f"New info: '{info_str}'. Saving to {save_path}")
    try:
        result = change_info(ckpt_path, info_str, final_save_name) # Передаем только имя файла
        logger.info(result)
    except Exception as e:
        logger.error(f"Changing info failed: {e}")
        logger.error(traceback.format_exc())

def show_info_cli(ckpt_path):
    """CLI wrapper for show_info."""
    logger.info(f"Showing info for {ckpt_path}")
    try:
        result = show_info(ckpt_path)
        print(f"--- Model Info: {ckpt_path} ---\n{result}\n-----------------------------")
    except Exception as e:
        logger.error(f"Showing info failed: {e}")
        logger.error(traceback.format_exc())

def extract_small_model_cli(ckpt_path, save_name, sr_str, if_f0_str, info_str, version):
    """CLI wrapper for extract_small_model."""
    global config # Используем глобальный config
    if not save_name:
        logger.error("Save name for extracted model is required (--save_name).")
        return
    if if_f0_str not in ["0", "1"]:
         logger.error("Invalid value for --if_f0. Must be '0' or '1'.")
         return
    save_path = os.path.join(weight_root, f"{save_name}.pth") # Полный путь
    logger.info(f"Extracting small model from {ckpt_path} to {save_path}")
    try:
        # Передаем if_f0_str как строку
        result = extract_small_model(ckpt_path, save_name, sr_str, if_f0_str, info_str, version)
        logger.info(result)
    except Exception as e:
        logger.error(f"Extracting small model failed: {e}")
        logger.error(traceback.format_exc())

# --- UVR5 function needs CLI wrapper ---
def uvr_cli(model_name, input_paths, output_dir_vocals, output_dir_instrumental, agg, format):
    """CLI wrapper for uvr."""
    global config # Используем глобальный config
    if not model_name in uvr5_names:
         # Проверяем наличие модели, если не найдено в списке
         model_file_path = os.path.join(weight_uvr5_root, f"{model_name}.pth")
         onnx_file_path = os.path.join(weight_uvr5_root, f"{model_name}.onnx")
         if not os.path.exists(model_file_path) and not os.path.exists(onnx_file_path):
             logger.error(f"Invalid UVR5 model name: {model_name}. Not found in known list or in {weight_uvr5_root}. Available: {uvr5_names}")
             return
         else:
             logger.info(f"Using custom UVR5 model: {model_name}")

    if not input_paths:
         logger.error("No input files or directory provided for UVR5.")
         return

    # Determine if input is a directory or list of files
    if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
        dir_wav_input = input_paths[0]
        # Получаем список файлов из директории
        try:
            paths = [os.path.join(dir_wav_input, name) for name in os.listdir(dir_wav_input) if name.lower().endswith(('.wav', '.flac', '.mp3', '.m4a'))]
            if not paths:
                logger.error(f"No audio files found in directory: {dir_wav_input}")
                return
            logger.info(f"Processing directory: {dir_wav_input}")
        except Exception as e:
             logger.error(f"Could not read directory {dir_wav_input}: {e}")
             return
    else:
        dir_wav_input = None # Не передаем директорию, если дан список файлов
        # Проверяем существование всех файлов в списке
        paths = []
        all_files_exist = True
        for p in input_paths:
            if os.path.isfile(p):
                paths.append(p)
            else:
                logger.error(f"Input file not found: {p}")
                all_files_exist = False
        if not all_files_exist:
            return
        logger.info(f"Processing files: {paths}")

    # Создаем выходные директории
    os.makedirs(output_dir_vocals, exist_ok=True)
    os.makedirs(output_dir_instrumental, exist_ok=True)

    logger.info(f"Starting UVR5 processing with model {model_name}...")
    try:
        # Передаем None для inp_root, если работаем со списком файлов
        # `uvr` должен быть модифицирован для обработки этого случая
        # или мы передаем `dir_wav_input` и `paths` (где paths - полные пути из dir_wav_input)
        uvr_generator = uvr(model_name, dir_wav_input if dir_wav_input else "", output_dir_vocals, paths, output_dir_instrumental, agg, format)
        # Обрабатываем вывод генератора
        final_message = "UVR5 processing finished."
        for message in uvr_generator:
            # Можно печатать промежуточные сообщения
            # print(message)
            final_message = message # Сохраняем последнее сообщение
        logger.info(final_message)

    except Exception as e:
        logger.error(f"UVR processing failed: {e}")
        logger.error(traceback.format_exc())


# --- Inference functions need CLI wrappers ---
def infer_one_cli(sid, input_audio, output_audio, transpose, f0_method,
                  index_path1, index_path2, index_rate, filter_radius,
                  resample_sr, rms_mix_rate, protect, f0_file=None):
    """CLI wrapper for single inference."""
    global vc, config # Используем глобальные vc и config
    if not vc:
        logger.error("VC object not initialized. Cannot infer.")
        return
    if not sid:
        logger.error("Speaker/model name (--sid) is required.")
        return
    # Проверяем существование входного файла
    if not os.path.isfile(input_audio):
        logger.error(f"Input audio file not found: {input_audio}")
        return

    if not output_audio:
        logger.error("Output audio path (--output_audio) is required.")
        return

    # Determine index path
    final_index_path = index_path1 if index_path1 else index_path2
    if not final_index_path:
        logger.warning("No index file provided or selected. Retrieval quality may be affected.")
    else:
        # Проверяем существование файла индекса
        if not os.path.exists(final_index_path):
             logger.error(f"Index file not found: {final_index_path}")
             return

    # Get speaker ID and potentially update protect based on model
    spk_id = 0 # ID говорящего по умолчанию
    final_protect = protect # Protect по умолчанию
    try:
        logger.info(f"Loading model: {sid}")
        # Получаем информацию и возможно обновляем protect
        get_vc_result = vc.get_vc(sid, protect, protect) # Передаем protect как в UI

        # Обрабатываем возможные форматы возвращаемого значения get_vc
        spk_item_update = None
        if isinstance(get_vc_result, tuple) and len(get_vc_result) >= 5:
            spk_item_update = get_vc_result[0]
            protect0_update = get_vc_result[1]
            # final_protect = protect0_update.get("value", protect) if isinstance(protect0_update, dict) else protect
            # Исправлено: используем protect1_update для protect, как в batch, для консистентности
            protect1_update = get_vc_result[2]
            final_protect = protect1_update.get("value", protect) if isinstance(protect1_update, dict) else protect
        elif isinstance(get_vc_result, dict) and 'maximum' in get_vc_result:
            spk_item_update = get_vc_result
            final_protect = protect
        else:
             logger.warning("Could not determine speaker ID or protect value from get_vc. Using defaults.")

        # Извлекаем ID говорящего
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
        return

    # Call the core inference function
    logger.info(f"Starting inference for {input_audio}...")
    try:
        info_text, audio_out = vc.vc_single(
            spk_id, # Use loaded speaker ID
            input_audio,
            transpose,
            f0_file,
            f0_method,
            final_index_path, # Pass the determined index path here
            "", # file_index2 - not used when path is provided
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            final_protect # Use protect value potentially updated by get_vc
        )
        logger.info(f"Inference Info: {info_text}")
        if audio_out and isinstance(audio_out, tuple) and len(audio_out) == 2:
             sr_out, audio_data = audio_out
             # Save the audio data
             import soundfile as sf # Импорт здесь, чтобы не требовался если не используется
             try:
                 # Убедимся, что директория для вывода существует
                 os.makedirs(os.path.dirname(output_audio), exist_ok=True)
                 sf.write(output_audio, audio_data, sr_out)
                 logger.info(f"Output audio saved to {output_audio} (Sample Rate: {sr_out})")
             except Exception as e:
                  logger.error(f"Failed to save output audio to {output_audio}: {e}")
        elif audio_out is None:
            logger.error(f"Inference failed. Check logs for details. Info: {info_text}")
        else:
            logger.warning(f"Inference completed but no valid audio output was generated. Info: {info_text}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())

def infer_batch_cli(sid, input_paths, output_dir, transpose, f0_method,
                   index_path1, index_path2, index_rate, filter_radius,
                   resample_sr, rms_mix_rate, protect, format):
    """CLI wrapper for batch inference."""
    global vc, config # Используем глобальные vc и config
    if not vc:
        logger.error("VC object not initialized. Cannot infer.")
        return
    if not sid:
        logger.error("Speaker/model name (--sid) is required.")
        return
    if not input_paths:
        logger.error("No input files or directory provided for batch inference.")
        return

    # Determine index path
    final_index_path = index_path1 if index_path1 else index_path2
    if not final_index_path:
        logger.warning("No index file provided or selected. Retrieval quality may be affected.")
    else:
        # Проверяем существование файла индекса
        if not os.path.exists(final_index_path):
             logger.error(f"Index file not found: {final_index_path}")
             return

    # Determine if input is a directory or list of files
    if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
        dir_input = input_paths[0]
        # Проверяем существование директории
        if not os.path.exists(dir_input):
            logger.error(f"Input directory not found: {dir_input}")
            return
        paths = None # vc_multi expects None if dir_input is used
        logger.info(f"Processing directory: {dir_input}")
    else:
        dir_input = None # vc_multi expects None if paths is used
        # Проверяем существование всех файлов в списке
        paths = []
        all_files_exist = True
        for p in input_paths:
            if os.path.isfile(p):
                paths.append(p)
            else:
                logger.error(f"Input file not found: {p}")
                all_files_exist = False
        if not all_files_exist:
            return
        logger.info(f"Processing files: {paths}")

    # Get speaker ID and potentially update protect based on model (similar to infer_one)
    spk_id = 0 # ID говорящего по умолчанию
    final_protect = protect # Protect по умолчанию
    try:
        logger.info(f"Loading model: {sid}")
        get_vc_result = vc.get_vc(sid, protect, protect) # Передаем protect как в UI для batch

        # Обрабатываем возможные форматы возвращаемого значения get_vc
        spk_item_update = None
        if isinstance(get_vc_result, tuple) and len(get_vc_result) >= 5:
            spk_item_update = get_vc_result[0]
            protect0_update = get_vc_result[1]
            protect1_update = get_vc_result[2] # Используем protect1 для batch как в UI
            index1_update = get_vc_result[3]
            index2_update = get_vc_result[4]
             # Используем обновленный protect1 если доступен
            final_protect = protect1_update.get("value", protect) if isinstance(protect1_update, dict) else protect

        elif isinstance(get_vc_result, dict) and 'maximum' in get_vc_result:
            spk_item_update = get_vc_result # Если возвращает только spk_item_update
            final_protect = protect
        else:
             logger.warning("Could not determine speaker ID or protect value from get_vc. Using defaults.")

         # Извлекаем ID говорящего (так же, как в infer_one_cli)
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
        return

    # Создаем выходную директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting batch inference...")
    try:
        # vc_multi теперь возвращает строку состояния
        info_generator = vc.vc_multi(
            spk_id,
            dir_input, # Pass directory path or None
            output_dir,
            paths, # Pass list of file paths or None
            transpose,
            f0_method,
            final_index_path, # Pass determined index path
            "", # file_index2 - not used when path is provided
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            final_protect, # Use protect value potentially updated by get_vc
            format
        )
        # Обрабатываем генератор (если vc_multi остался генератором)
        final_message = "Batch inference finished."
        # Выводим сообщения генератора в лог
        for message in info_generator:
            # logger.info(f"Batch Progress: {message}") # Можно раскомментировать для детального лога
            final_message = message # Сохраняем последнее сообщение

        logger.info(final_message) # Выводим последнее сообщение
        logger.info(f"Batch processing finished. Check output directory: '{output_dir}'")

    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        logger.error(traceback.format_exc())

def export_onnx_cli(ckpt_path, onnx_path):
    """CLI wrapper for ONNX export."""
    logger.info(f"Exporting {ckpt_path} to {onnx_path}")
    # Убедимся, что директория для вывода существует
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    try:
        from infer.modules.onnx.export import export_onnx as eo
        result = eo(ckpt_path, onnx_path)
        logger.info(f"ONNX export result: {result}")
    except ImportError:
        logger.error("ONNX export dependencies not found. Please ensure necessary libraries are installed.")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        logger.error(traceback.format_exc())


def main():
    global config, vc # Указываем, что будем использовать глобальные переменные

    # --- Parse CLI arguments ---
    parser = argparse.ArgumentParser(description="RVC CLI Tool", add_help=True) # add_help=True
    # Добавляем аргументы, которые раньше парсились в Config.__init__
    # Они будут проигнорированы Config, но могут быть полезны для других целей или логгирования
    #parser.add_argument("--pycmd", type=str, help="Python command (overrides config)")
    #parser.add_argument("--colab", action="store_true", help="Flag for Colab environment (overrides config)")
    #parser.add_argument("--noparallel", action="store_true", help="Disable parallel processing (overrides config)")
    #parser.add_argument("--dml", action="store_true", help="Enable DirectML (experimental, overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., cpu, cuda:0, mps, xpu:0). Overrides auto-detection.")
    parser.add_argument("--half", action='store_true', help="Force half precision (if available). Overrides auto-detection.")
    parser.add_argument("--nohalf", action='store_true', help="Force full precision. Overrides auto-detection.")


    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=False) # required=False для корневых

    # --- Preprocess Command ---
    parser_preprocess = subparsers.add_parser('preprocess', help='Preprocess audio dataset')
    parser_preprocess.add_argument('--trainset_dir', type=str, required=True, help='Path to the training dataset directory')
    parser_preprocess.add_argument('--exp_dir', type=str, required=True, help='Experiment directory name (will be created under ./logs)')
    parser_preprocess.add_argument('--sr', type=str, required=True, choices=list(sr_dict.keys()), help='Target sample rate')
    parser_preprocess.add_argument('--n_cpu', type=int, default=None, help='Number of CPU processes (default: auto based on config)')
    parser_preprocess.set_defaults(func=lambda args: preprocess_dataset_cli(args.trainset_dir, args.exp_dir, args.sr, args.n_cpu if args.n_cpu is not None else int(np.ceil(config.n_cpu / 1.5))))

    # --- Extract Features Command ---
    parser_extract = subparsers.add_parser('extract', help='Extract features (and optionally F0 pitch)')
    parser_extract.add_argument('--exp_dir', type=str, required=True, help='Experiment directory name (must exist under ./logs)')
    parser_extract.add_argument('--gpus', type=str, default=gpus, help='GPU IDs for feature extraction, separated by "-", e.g., "0-1"')
    parser_extract.add_argument('--n_cpu', type=int, default=None, help='Number of CPU processes for F0 extraction (default: auto based on config)')
    parser_extract.add_argument('--f0', action='store_true', help='Enable F0 pitch extraction')
    parser_extract.add_argument('--f0_method', type=str, default='rmvpe_gpu', choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"], help='F0 extraction method')
    parser_extract.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'], help='Model version (affects feature dimension)')
    parser_extract.add_argument('--gpus_rmvpe', type=str, default=None, help='GPU IDs for RMVPE F0 extraction, e.g., "0-0-1" (default: use --gpus argument)')
    parser_extract.set_defaults(func=lambda args: extract_f0_feature_cli(args.gpus, args.n_cpu if args.n_cpu is not None else int(np.ceil(config.n_cpu / 1.5)), args.f0_method, args.f0, args.exp_dir, args.version, args.gpus_rmvpe if args.gpus_rmvpe is not None else args.gpus)) # Default gpus_rmvpe to gpus

    # --- Train Model Command ---
    parser_train = subparsers.add_parser('train', help='Train the RVC model')
    parser_train.add_argument('--exp_dir', type=str, required=True, help='Experiment directory name')
    parser_train.add_argument('--sr', type=str, required=True, choices=list(sr_dict.keys()), help='Target sample rate')
    parser_train.add_argument('--f0', action='store_true', help='Whether the model uses F0 pitch')
    parser_train.add_argument('--spk_id', type=int, default=0, help='Speaker ID for training')
    parser_train.add_argument('--save_epoch', type=int, default=5, help='Save frequency (every N epochs)')
    parser_train.add_argument('--total_epoch', type=int, default=20, help='Total training epochs')
    parser_train.add_argument('--batch_size', type=int, default=None, help=f'Batch size per GPU (default: auto based on GPU memory, currently {default_batch_size})')
    parser_train.add_argument('--save_latest', action='store_true', help='Only save the latest checkpoint')
    parser_train.add_argument('--pretrained_g', type=str, default="", help='Path to pretrained Generator model (G). Auto-detect if empty.')
    parser_train.add_argument('--pretrained_d', type=str, default="", help='Path to pretrained Discriminator model (D). Auto-detect if empty.')
    parser_train.add_argument('--gpus', type=str, default=gpus, help='GPU IDs for training, separated by "-", e.g., "0-1"')
    parser_train.add_argument('--cache_gpu', action='store_true', help='Cache dataset to GPU memory (for small datasets)')
    parser_train.add_argument('--save_weights', action='store_true', help='Save final small models to weights folder every save epoch')
    parser_train.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'], help='Model version')
    parser_train.set_defaults(func=lambda args: click_train_cli(args.exp_dir, args.sr, args.f0, args.spk_id, args.save_epoch, args.total_epoch, args.batch_size if args.batch_size is not None else default_batch_size, args.save_latest, args.pretrained_g, args.pretrained_d, args.gpus, args.cache_gpu, args.save_weights, args.version))

    # --- Train Index Command ---
    parser_index = subparsers.add_parser('index', help='Train the FAISS index')
    parser_index.add_argument('--exp_dir', type=str, required=True, help='Experiment directory name')
    parser_index.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'], help='Model version (affects feature dimension)')
    parser_index.set_defaults(func=lambda args: train_index_cli(args.exp_dir, args.version))

    # --- Train All Command ---
    parser_train_all = subparsers.add_parser('train-all', help='Run preprocessing, feature extraction, training, and indexing sequentially')
    parser_train_all.add_argument('--exp_dir', type=str, required=True, help='Experiment directory name')
    parser_train_all.add_argument('--trainset_dir', type=str, required=True, help='Path to the training dataset directory')
    parser_train_all.add_argument('--sr', type=str, required=True, choices=list(sr_dict.keys()), help='Target sample rate')
    parser_train_all.add_argument('--f0', action='store_true', help='Enable F0 pitch extraction and usage')
    parser_train_all.add_argument('--spk_id', type=int, default=0, help='Speaker ID for training')
    parser_train_all.add_argument('--n_cpu', type=int, default=None, help='Number of CPU processes (default: auto)')
    parser_train_all.add_argument('--f0_method', type=str, default='rmvpe_gpu', choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"], help='F0 extraction method')
    parser_train_all.add_argument('--save_epoch', type=int, default=5, help='Save frequency (every N epochs)')
    parser_train_all.add_argument('--total_epoch', type=int, default=20, help='Total training epochs')
    parser_train_all.add_argument('--batch_size', type=int, default=None, help=f'Batch size per GPU (default: auto, {default_batch_size})')
    parser_train_all.add_argument('--save_latest', action='store_true', help='Only save the latest checkpoint')
    parser_train_all.add_argument('--pretrained_g', type=str, default="", help='Path to pretrained G model (auto-detect if empty)')
    parser_train_all.add_argument('--pretrained_d', type=str, default="", help='Path to pretrained D model (auto-detect if empty)')
    parser_train_all.add_argument('--gpus', type=str, default=gpus, help='GPU IDs for training/extraction, separated by "-", e.g., "0-1"')
    parser_train_all.add_argument('--cache_gpu', action='store_true', help='Cache dataset to GPU memory')
    parser_train_all.add_argument('--save_weights', action='store_true', help='Save final small models to weights folder')
    parser_train_all.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'], help='Model version')
    parser_train_all.add_argument('--gpus_rmvpe', type=str, default=None, help='GPU IDs for RMVPE F0 extraction, e.g., "0-0-1" (default: same as --gpus)')
    parser_train_all.set_defaults(func=lambda args: train1key_cli(args.exp_dir, args.sr, args.f0, args.trainset_dir, args.spk_id, args.n_cpu if args.n_cpu is not None else int(np.ceil(config.n_cpu / 1.5)), args.f0_method, args.save_epoch, args.total_epoch, args.batch_size if args.batch_size is not None else default_batch_size, args.save_latest, args.pretrained_g, args.pretrained_d, args.gpus, args.cache_gpu, args.save_weights, args.version, args.gpus_rmvpe if args.gpus_rmvpe is not None else args.gpus)) # Default gpus_rmvpe to gpus


    # --- Infer One Command ---
    parser_infer_one = subparsers.add_parser('infer-one', help='Perform single audio file inference')
    parser_infer_one.add_argument('--sid', type=str, required=True, help='Speaker model name (e.g., myvoice.pth from assets/weights)')
    parser_infer_one.add_argument('--input_audio', type=str, required=True, help='Path to the input audio file')
    parser_infer_one.add_argument('--output_audio', type=str, required=True, help='Path to save the output audio file')
    parser_infer_one.add_argument('--transpose', type=int, default=0, help='Transpose (pitch shift) in semitones')
    parser_infer_one.add_argument('--f0_method', type=str, default='rmvpe', choices=["pm", "harvest", "crepe", "rmvpe"], help='F0 prediction method')
    parser_infer_one.add_argument('--index_path', type=str, default=None, help='Path to the FAISS index file (e.g., logs/myvoice/added_...index)')
    # --index_path2 removed
    parser_infer_one.add_argument('--index_rate', type=float, default=0.75, help='Feature retrieval ratio (0 to 1)')
    parser_infer_one.add_argument('--filter_radius', type=int, default=3, help='Median filter radius for Harvest F0 (>=3 applies filter)')
    parser_infer_one.add_argument('--resample_sr', type=int, default=0, help='Resample output audio to this SR (0 for no resampling)')
    parser_infer_one.add_argument('--rms_mix_rate', type=float, default=0.25, help='RMS mix rate (0 to 1)')
    parser_infer_one.add_argument('--protect', type=float, default=0.33, help='Protection for consonants/breaths (0 to 0.5)')
    parser_infer_one.add_argument('--f0_file', type=str, default=None, help='Optional external F0 file path')
    parser_infer_one.set_defaults(func=lambda args: infer_one_cli(args.sid, args.input_audio, args.output_audio, args.transpose, args.f0_method, args.index_path, None, args.index_rate, args.filter_radius, args.resample_sr, args.rms_mix_rate, args.protect, args.f0_file))

    # --- Infer Batch Command ---
    parser_infer_batch = subparsers.add_parser('infer-batch', help='Perform batch inference on multiple audio files or a directory')
    parser_infer_batch.add_argument('--sid', type=str, required=True, help='Speaker model name (e.g., myvoice.pth)')
    parser_infer_batch.add_argument('--input_paths', type=str, nargs='+', required=True, help='Paths to input audio files OR a single directory path')
    parser_infer_batch.add_argument('--output_dir', type=str, default='output', help='Directory to save output audio files')
    parser_infer_batch.add_argument('--transpose', type=int, default=0, help='Transpose (pitch shift) in semitones')
    parser_infer_batch.add_argument('--f0_method', type=str, default='rmvpe', choices=["pm", "harvest", "crepe", "rmvpe"], help='F0 prediction method')
    parser_infer_batch.add_argument('--index_path', type=str, default=None, help='Path to the FAISS index file')
    # --index_path2 removed
    parser_infer_batch.add_argument('--index_rate', type=float, default=1.0, help='Feature retrieval ratio (0 to 1)') # Default was 1 for batch
    parser_infer_batch.add_argument('--filter_radius', type=int, default=3, help='Median filter radius for Harvest F0 (>=3 applies filter)')
    parser_infer_batch.add_argument('--resample_sr', type=int, default=0, help='Resample output audio to this SR (0 for no resampling)')
    parser_infer_batch.add_argument('--rms_mix_rate', type=float, default=1.0, help='RMS mix rate (0 to 1)') # Default was 1 for batch
    parser_infer_batch.add_argument('--protect', type=float, default=0.33, help='Protection for consonants/breaths (0 to 0.5)')
    parser_infer_batch.add_argument('--format', type=str, default='wav', choices=['wav', 'flac', 'mp3', 'm4a'], help='Output audio format')
    parser_infer_batch.set_defaults(func=lambda args: infer_batch_cli(args.sid, args.input_paths, args.output_dir, args.transpose, args.f0_method, args.index_path, None, args.index_rate, args.filter_radius, args.resample_sr, args.rms_mix_rate, args.protect, args.format))

    # --- UVR5 Command ---
    parser_uvr = subparsers.add_parser('uvr', help='Separate vocals and instrumentals using UVR5')
    parser_uvr.add_argument('--model_name', type=str, required=False, # Сделаем не обязательным для list
                            choices=uvr5_names + ["list"] if uvr5_names else ["list"], # Add "list" option
                            help=f'UVR5 model name or "list" to show available models.')
    parser_uvr.add_argument('--input_paths', type=str, nargs='+', required=False, help='Paths to input audio files OR a single directory path (required unless listing models)')
    parser_uvr.add_argument('--output_dir_vocals', type=str, default='output/vocals', help='Directory to save vocal tracks')
    parser_uvr.add_argument('--output_dir_instrumental', type=str, default='output/instrumental', help='Directory to save instrumental tracks')
    parser_uvr.add_argument('--agg', type=int, default=10, help='Aggressiveness for vocal extraction (0-20)')
    parser_uvr.add_argument('--format', type=str, default='flac', choices=['wav', 'flac', 'mp3', 'm4a'], help='Output audio format')
    parser_uvr.set_defaults(func=lambda args: uvr_cli(args.model_name, args.input_paths, args.output_dir_vocals, args.output_dir_instrumental, args.agg, args.format))


    # --- Ckpt Merge Command ---
    parser_ckpt_merge = subparsers.add_parser('ckpt-merge', help='Merge two RVC models')
    parser_ckpt_merge.add_argument('--ckpt_a', type=str, required=True, help='Path to model A (.pth)')
    parser_ckpt_merge.add_argument('--ckpt_b', type=str, required=True, help='Path to model B (.pth)')
    parser_ckpt_merge.add_argument('--alpha', type=float, default=0.5, help='Weight of model A (0 to 1)')
    parser_ckpt_merge.add_argument('--sr', type=str, required=True, choices=list(sr_dict.keys()), help='Target sample rate for merged model')
    parser_ckpt_merge.add_argument('--f0', action='store_true', help='Set if merged model should use F0 pitch')
    parser_ckpt_merge.add_argument('--info', type=str, default="", help='Optional info string for merged model metadata')
    parser_ckpt_merge.add_argument('--save_name', type=str, required=True, help='Filename (without .pth) to save the merged model (in assets/weights)')
    parser_ckpt_merge.add_argument('--version', type=str, required=True, choices=['v1', 'v2'], help='Target version for merged model')
    parser_ckpt_merge.set_defaults(func=lambda args: merge_cli(args.ckpt_a, args.ckpt_b, args.alpha, args.sr, args.f0, args.info, args.save_name, args.version))

    # --- Ckpt Modify Command ---
    parser_ckpt_modify = subparsers.add_parser('ckpt-modify', help='Modify metadata info string of an RVC model')
    parser_ckpt_modify.add_argument('--ckpt_path', type=str, required=True, help='Path to the model (.pth) to modify')
    parser_ckpt_modify.add_argument('--info', type=str, required=True, help='New info string for model metadata')
    parser_ckpt_modify.add_argument('--save_name', type=str, default="", help='Optional: New filename (without .pth). Overwrites if empty.')
    parser_ckpt_modify.set_defaults(func=lambda args: change_info_cli(args.ckpt_path, args.info, args.save_name))

    # --- Ckpt Show Command ---
    parser_ckpt_show = subparsers.add_parser('ckpt-show', help='Show metadata info of an RVC model')
    parser_ckpt_show.add_argument('--ckpt_path', type=str, required=True, help='Path to the model (.pth)')
    parser_ckpt_show.set_defaults(func=lambda args: show_info_cli(args.ckpt_path))

    # --- Ckpt Extract Command ---
    parser_ckpt_extract = subparsers.add_parser('ckpt-extract', help='Extract a small inference model from a large training checkpoint')
    parser_ckpt_extract.add_argument('--ckpt_path', type=str, required=True, help='Path to the large training checkpoint (e.g., G_xxxxx.pth)')
    parser_ckpt_extract.add_argument('--save_name', type=str, required=True, help='Filename (without .pth) to save the extracted model (in assets/weights)')
    parser_ckpt_extract.add_argument('--sr', type=str, required=True, choices=list(sr_dict.keys()), help='Target sample rate')
    parser_ckpt_extract.add_argument('--if_f0', type=str, required=True, choices=['0', '1'], help='Whether the model uses F0 pitch (1 for yes, 0 for no)')
    parser_ckpt_extract.add_argument('--info', type=str, default="", help='Optional info string for extracted model metadata')
    parser_ckpt_extract.add_argument('--version', type=str, required=True, choices=['v1', 'v2'], help='Model version')
    parser_ckpt_extract.set_defaults(func=lambda args: extract_small_model_cli(args.ckpt_path, args.save_name, args.sr, args.if_f0, args.info, args.version))

    # --- Export ONNX Command ---
    parser_export_onnx = subparsers.add_parser('export-onnx', help='Export an RVC model to ONNX format')
    parser_export_onnx.add_argument('--ckpt_path', type=str, required=True, help='Path to the RVC model (.pth)')
    parser_export_onnx.add_argument('--onnx_path', type=str, required=True, help='Path to save the output ONNX model')
    parser_export_onnx.set_defaults(func=lambda args: export_onnx_cli(args.ckpt_path, args.onnx_path))


    # --- Parse arguments and execute function ---
    args = parser.parse_args()

    # Инициализируем Config *после* парсинга корневых аргументов, но *перед* вызовом функции команды
    config = Config() # Config теперь синглтон
    # Переопределяем device и is_half из Config, если они были заданы в командной строке
    if args.device is not None:
        logger.info(f"Overriding device with CLI argument: {args.device}")
        config.device = args.device
        # Нужно перезапустить device_config, если device был переопределен
        # чтобы is_half и другие параметры соответствовали
        config.device_config() # Переконфигурируем устройство

    if args.half and args.nohalf:
        logger.warning("Both --half and --nohalf specified, using --nohalf (full precision).")
        config.is_half = False
    elif args.half:
         if not config.is_half: logger.info("Overriding precision with CLI argument: --half")
         config.is_half = True
    elif args.nohalf:
         if config.is_half: logger.info("Overriding precision with CLI argument: --nohalf")
         config.is_half = False
    # Применяем изменения DML и pycmd из корневых аргументов, если они были переданы
    # if hasattr(args, 'dml'): config.dml = args.dml # DML уже обрабатывается в Config
    # if hasattr(args, 'pycmd'): config.python_cmd = args.pycmd # pycmd уже обрабатывается в Config
    # if hasattr(args, 'noparallel'): config.noparallel = args.noparallel # noparallel уже обрабатывается в Config

    # Инициализация VC после Config
    vc = VC(config)

    # Проверка, была ли передана команда
    if args.command is None:
         # Если нет команды, возможно, это запуск UI - не выводим ошибку
         # Если предполагается только CLI, то:
         parser.print_help(sys.stderr)
         sys.exit("Error: No command provided. Please choose one of the available commands.")

    # Special handling for UVR list models
    if args.command == 'uvr':
        if args.model_name == 'list':
            print("\nAvailable UVR5 models:")
            if uvr5_names:
                for name in sorted(uvr5_names):
                    print(f"- {name}")
            else:
                print("  No models found (check assets/uvr5_weights).")
            print("\nUse the 'uvr' command with '--model_name <model>' and input/output paths to run separation.")
            sys.exit(0)
        elif not args.model_name: # Если не 'list' и не задано имя
             parser_uvr.error("argument --model_name is required (unless using 'list')")
        elif not args.input_paths:
             parser_uvr.error("argument --input_paths is required when specifying a model")


    # Вызов функции, связанной с подкомандой
    args.func(args)

if __name__ == "__main__":
    main()