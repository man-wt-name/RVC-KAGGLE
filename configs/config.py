import argparse
import os
import sys
import json
import shutil
from multiprocessing import cpu_count

import torch

# --- ДОБАВЛЕНО: Определение now_dir ---
# Предполагается, что этот скрипт выполняется из корневой директории проекта
now_dir = os.getcwd()
# -----------------------------------

try:
    # import intel_extension_for_pytorch as ipex # Закомментировано, если не используется IPEX
    # if torch.xpu.is_available():
    #     from infer.modules.ipex import ipex_init
    #     ipex_init()
    pass # Оставляем pass, если блок try/except нужен для других целей
except Exception:
    pass
import logging

logger = logging.getLogger(__name__)


version_config_list = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, 'instance') or wrapper.instance is None:
             wrapper.instance = func(*args, **kwargs)
        # Убрано обнуление instance = None, чтобы config был действительно синглтоном
        # wrapper.instance = None
        return wrapper.instance

    # wrapper.instance = None # Инициализация убрана из декоратора
    return wrapper


@singleton_variable
class Config:
    def __init__(self):
        # Проверяем, был ли уже инициализирован синглтон
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.device = "cuda:0"
        self.is_half = True
        self.use_jit = False
        self.n_cpu = 0
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.dml,
        ) = self.arg_parse() # Этот вызов остается, но он будет игнорировать 'train-all'
        self.instead = ""
        self.preprocess_per = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

        self._initialized = True # Флаг инициализации

    @staticmethod
    def load_config_json() -> dict:
        d = {}
        base_configs_dir = os.path.join(now_dir, "configs") # Используем now_dir
        inuse_configs_dir = os.path.join(base_configs_dir, "inuse")
        os.makedirs(inuse_configs_dir, exist_ok=True) # Убедиться, что директория существует

        for config_file_rel in version_config_list: # e.g., "v1/32k.json"
            base_config_path = os.path.join(base_configs_dir, config_file_rel)
            inuse_config_path = os.path.join(inuse_configs_dir, config_file_rel)
            inuse_config_subdir = os.path.dirname(inuse_config_path)

            # Создать поддиректорию в inuse, если её нет
            os.makedirs(inuse_config_subdir, exist_ok=True)

            if not os.path.exists(inuse_config_path):
                if not os.path.exists(base_config_path):
                    logger.warning(f"Base config file '{base_config_path}' not found. Skipping copy.")
                    continue # Пропустить, если базовый конфиг не найден
                try:
                    shutil.copy(base_config_path, inuse_config_path)
                    logger.info(f"Copied '{base_config_path}' to '{inuse_config_path}'")
                except Exception as e:
                    logger.error(f"Error copying config file '{base_config_path}' to '{inuse_config_path}': {e}")
                    continue # Пропустить, если копирование не удалось

            # Проверить существование файла снова после попытки копирования
            if os.path.exists(inuse_config_path):
                 try:
                     with open(inuse_config_path, "r", encoding="utf-8") as f: # Добавлено utf-8
                         d[config_file_rel] = json.load(f)
                 except Exception as e:
                     logger.error(f"Error loading config file '{inuse_config_path}': {e}")
            else:
                 logger.warning(f"Config file '{inuse_config_path}' not found and could not be created. Skipping loading.")

        return d


    @staticmethod
    def arg_parse() -> tuple:
        # Используем add_help=False, чтобы избежать конфликта с основным парсером
        parser = argparse.ArgumentParser(add_help=False)
        # Определяем только те аргументы, которые относятся к Config
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=sys.executable or "python", help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument("--noparallel", action="store_true", help="Disable parallel processing")
        parser.add_argument("--noautoopen", action="store_true", help="Do not open in browser automatically")
        parser.add_argument("--dml", action="store_true", help="torch_dml")

        # Используем parse_known_args(), чтобы игнорировать неизвестные аргументы
        cmd_opts, _ = parser.parse_known_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        # Возвращаем значения по умолчанию, если они не были переопределены
        pycmd = cmd_opts.pycmd if hasattr(cmd_opts, 'pycmd') else sys.executable or "python"
        port = cmd_opts.port if hasattr(cmd_opts, 'port') else 7865
        colab = cmd_opts.colab if hasattr(cmd_opts, 'colab') else False
        noparallel = cmd_opts.noparallel if hasattr(cmd_opts, 'noparallel') else False
        noautoopen = cmd_opts.noautoopen if hasattr(cmd_opts, 'noautoopen') else False
        dml = cmd_opts.dml if hasattr(cmd_opts, 'dml') else False


        return (
            pycmd,
            port,
            colab,
            noparallel,
            noautoopen,
            dml,
        )

    @staticmethod
    def has_mps() -> bool:
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
             return False
        try:
            # Проверка на macOS для дополнительной надежности
            if sys.platform != "darwin":
                return False
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @staticmethod
    def has_xpu() -> bool:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False

    def use_fp32_config(self):
        # Убедимся, что self.json_config инициализирован
        if not hasattr(self, 'json_config') or not self.json_config:
            logger.warning("json_config not initialized in Config. Skipping use_fp32_config.")
            return

        base_configs_dir = os.path.join(now_dir, "configs") # Используем now_dir
        inuse_configs_dir = os.path.join(base_configs_dir, "inuse")

        for config_file_rel in version_config_list:
             config_path_inuse = os.path.join(inuse_configs_dir, config_file_rel)
             if os.path.exists(config_path_inuse):
                 try:
                     # Убеждаемся, что ключ существует перед записью
                     if config_file_rel in self.json_config:
                        self.json_config[config_file_rel]["train"]["fp16_run"] = False
                     else:
                        logger.warning(f"Config key '{config_file_rel}' not found in json_config during use_fp32_config.")

                     with open(config_path_inuse, "r", encoding="utf-8") as f: # Добавлено utf-8
                         strr = f.read().replace("true", "false")
                     with open(config_path_inuse, "w", encoding="utf-8") as f: # Добавлено utf-8
                         f.write(strr)
                     logger.info(f"Overwritten '{config_path_inuse}' to use fp32.")
                 except KeyError as e:
                     logger.error(f"Key error processing config file {config_path_inuse} in use_fp32_config: {e}. Check JSON structure.")
                 except Exception as e:
                     logger.error(f"Error processing config file {config_path_inuse} in use_fp32_config: {e}")
             else:
                 logger.warning(f"Config file '{config_path_inuse}' not found during use_fp32_config. Skipping.")

        self.preprocess_per = 3.0 # Это должно быть вне цикла
        logger.info(f"Overwritten preprocess_per to {self.preprocess_per}")

    def determine_optimal_device(self):
        """Определяет наилучшее доступное устройство."""
        if torch.cuda.is_available():
            logger.debug("CUDA is available.")
            return "cuda:0" # По умолчанию используем первую CUDA карту
        elif self.has_mps():
            logger.debug("MPS is available.")
            return "mps"
        elif self.has_xpu():
             logger.debug("XPU is available.")
             return "xpu:0" # По умолчанию используем первую XPU карту
        else:
            logger.debug("No GPU found, using CPU.")
            return "cpu"

    def device_config(self) -> tuple:
        # Переопределяем device и is_half на основе доступности
        # Сначала определяем оптимальное устройство
        optimal_device = self.determine_optimal_device()

        # Если в __init__ был задан device через аргументы (хотя мы его убрали оттуда),
        # он мог бы быть здесь. Но сейчас мы его определяем здесь.
        self.device = optimal_device # Устанавливаем оптимальное устройство

        # Сбрасываем is_half перед проверками
        self.is_half = optimal_device not in ["cpu", "mps"] # По умолчанию half precision для GPU

        logger.info(f"Attempting to configure for device: {self.device}")

        if self.device.startswith("cuda"):
            try:
                i_device = int(self.device.split(":")[-1])
                if not (0 <= i_device < torch.cuda.device_count()):
                    logger.warning(f"Invalid CUDA device index {i_device}. Defaulting to 0.")
                    i_device = 0
                    self.device = "cuda:0"

                self.gpu_name = torch.cuda.get_device_name(i_device)
                # Проверка поддержки FP16 (некоторые старые карты могут не поддерживать)
                props = torch.cuda.get_device_properties(i_device)
                has_fp16 = props.major >= 7 # Compute capability 7.0+ generally supports fast FP16

                if not has_fp16 or (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "P10" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
                ):
                    force_fp32_reason = "does not support FP16" if not has_fp16 else "known to have slow FP16"
                    logger.info(f"Found GPU {self.gpu_name} which {force_fp32_reason}, forcing to fp32")
                    self.is_half = False
                    self.use_fp32_config()
                else:
                    logger.info("Found GPU %s", self.gpu_name)
                    # Оставляем is_half=True, если карта поддерживает

                self.gpu_mem = int(props.total_memory / 1024 / 1024 / 1024 + 0.4)
                if self.gpu_mem <= 4:
                    logger.warning(f"GPU memory {self.gpu_mem}GB <= 4GB. Setting preprocess_per=3.0.")
                    self.preprocess_per = 3.0
            except Exception as e:
                 logger.error(f"Error during CUDA device configuration: {e}. Falling back to CPU.")
                 self.device = "cpu"
                 self.is_half = False
                 self.use_fp32_config()

        elif self.device == "mps":
            logger.info("Configuring for MPS device.")
            self.is_half = False # MPS обычно не поддерживает half precision
            self.use_fp32_config()
        elif self.device.startswith("xpu"):
             logger.info("Configuring for XPU device.")
             # XPU может поддерживать is_half, оставляем True (установлено по умолчанию)
             # Если нужно форсировать FP32 для XPU, добавить:
             # self.is_half = False
             # self.use_fp32_config()
             # Получение информации о памяти для XPU может отличаться
             try:
                  props = torch.xpu.get_device_properties(self.device)
                  self.gpu_name = props.name
                  self.gpu_mem = int(props.total_memory / 1024 / 1024 / 1024 + 0.4)
                  # Дополнительные проверки для XPU, если нужны
             except Exception as e:
                  logger.warning(f"Could not get XPU properties: {e}")
                  self.gpu_name = "Unknown XPU"
                  self.gpu_mem = None # Неизвестно
        else: # CPU
            logger.info("No supported GPU found. Configuring for CPU.")
            self.device = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        # Настройка x_pad и других параметров на основе is_half и gpu_mem
        if self.is_half:
            x_pad = 3; x_query = 10; x_center = 60; x_max = 65
        else:
            x_pad = 1; x_query = 6; x_center = 38; x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            logger.warning(f"GPU memory {self.gpu_mem}GB <= 4GB. Adjusting x_pad parameters.")
            x_pad = 1; x_query = 5; x_center = 30; x_max = 32

        # --- DML logic - теперь использует now_dir ---
        runtime_path_prefix = os.path.join(now_dir, "runtime", "Lib", "site-packages")
        onnxruntime_cuda_path = os.path.join(runtime_path_prefix, "onnxruntime-cuda")
        onnxruntime_dml_path = os.path.join(runtime_path_prefix, "onnxruntime-dml")
        onnxruntime_active_path = os.path.join(runtime_path_prefix, "onnxruntime")

        if self.dml:
            logger.info("Use DirectML instead")
            dml_dll_path = os.path.join(onnxruntime_active_path, "capi", "DirectML.dll")
            if not os.path.exists(dml_dll_path):
                logger.info("Attempting to switch to DML ONNX runtime...")
                try:
                    if os.path.exists(onnxruntime_active_path) and not os.path.islink(onnxruntime_active_path): # Check if it's not already a link
                       backup_count = 1
                       target_rename_path = os.path.join(runtime_path_prefix, f"onnxruntime-backup-{backup_count}")
                       while os.path.exists(target_rename_path):
                           backup_count += 1
                           target_rename_path = os.path.join(runtime_path_prefix, f"onnxruntime-backup-{backup_count}")

                       os.rename(onnxruntime_active_path, target_rename_path)
                       logger.info(f"Renamed existing 'onnxruntime' to '{os.path.basename(target_rename_path)}'")
                except Exception as e:
                    logger.warning(f"Could not rename existing 'onnxruntime': {e}")

                try:
                    if os.path.exists(onnxruntime_dml_path):
                       os.rename(onnxruntime_dml_path, onnxruntime_active_path)
                       logger.info("Successfully renamed 'onnxruntime-dml' to 'onnxruntime'")
                    else:
                        logger.error("onnxruntime-dml directory not found, cannot switch.")
                except Exception as e:
                    logger.error(f"Could not rename 'onnxruntime-dml' to 'onnxruntime': {e}")

            # Check DML availability again
            if os.path.exists(dml_dll_path):
                try:
                     import torch_directml
                     dml_device = torch_directml.device(torch_directml.default_device())
                     logger.info(f"DirectML device detected: {dml_device}")
                     self.device = dml_device # !!! DML переопределяет device !!!
                     self.is_half = False
                     self.use_fp32_config() # DML usually requires fp32
                except ImportError:
                     logger.error("torch_directml not installed, cannot use DML device.")
                     if self.device != "cpu": # Не откатываться на CPU, если уже CPU
                          logger.warning("Falling back to CPU due to missing torch_directml.")
                          self.device = "cpu"
                          self.is_half = False
                          self.use_fp32_config()
                except Exception as e:
                    logger.error(f"Error setting up DirectML device: {e}")
                    if self.device != "cpu":
                         logger.warning("Falling back to CPU due to DirectML error.")
                         self.device = "cpu"
                         self.is_half = False
                         self.use_fp32_config()
            else:
                 logger.warning("DirectML.dll not found after attempting runtime switch. DML setup failed.")
                 # Не откатываемся на CPU здесь, возможно, пользователь все еще хочет CUDA/XPU/MPS

        else: # Not using DML, try CUDA for ONNX if needed
            if self.instead: # If XPU was detected earlier
                logger.info(f"Using {self.instead} for PyTorch backend.")
            cuda_dll_path = os.path.join(onnxruntime_active_path, "capi", "onnxruntime_providers_cuda.dll")
            if not os.path.exists(cuda_dll_path) and not self.device.startswith("xpu"): # Не переключаем, если используется XPU
                logger.info("Attempting to switch to CUDA ONNX runtime...")
                try:
                     if os.path.exists(onnxruntime_active_path) and not os.path.islink(onnxruntime_active_path):
                         backup_count = 1
                         target_rename_path = os.path.join(runtime_path_prefix, f"onnxruntime-backup-{backup_count}")
                         while os.path.exists(target_rename_path):
                             backup_count += 1
                             target_rename_path = os.path.join(runtime_path_prefix, f"onnxruntime-backup-{backup_count}")
                         os.rename(onnxruntime_active_path, target_rename_path)
                         logger.info(f"Renamed existing 'onnxruntime' to '{os.path.basename(target_rename_path)}'")
                except Exception as e:
                    logger.warning(f"Could not rename existing 'onnxruntime': {e}")

                try:
                    if os.path.exists(onnxruntime_cuda_path):
                        os.rename(onnxruntime_cuda_path, onnxruntime_active_path)
                        logger.info("Successfully renamed 'onnxruntime-cuda' to 'onnxruntime'")
                    else:
                         logger.warning("onnxruntime-cuda directory not found, cannot switch.")
                except Exception as e:
                    logger.error(f"Could not rename 'onnxruntime-cuda' to 'onnxruntime': {e}")
            # Final check after attempting switch
            if not os.path.exists(cuda_dll_path) and self.device.startswith("cuda"):
                 logger.warning("onnxruntime_providers_cuda.dll not found after attempting runtime switch. CUDA backend for ONNX might not work.")


        logger.info(
            "Final configuration: Half-precision=%s, Device=%s"
            % (self.is_half, self.device)
        )
        return x_pad, x_query, x_center, x_max