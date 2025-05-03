# run_rvc_cli.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
import platform
import torch
import numpy as np
from dotenv import load_dotenv

# Добавляем текущую директорию в путь, если скрипт запускается из корня проекта
now_dir = os.getcwd()
if now_dir not in sys.path:
     sys.path.append(now_dir)

load_dotenv()

# Импорт базовых классов и словаря sr_dict
from configs.config import Config
from infer.modules.vc.modules import VC
from i18n.i18n import I18nAuto

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO, # Уровень логирования (можно поставить DEBUG для большего количества сообщений)
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Вывод в консоль
        # Можно добавить FileHandler сюда же, если нужна и запись в файл
        # logging.FileHandler("cli_main.log")
    ]
)
logger = logging.getLogger(__name__)

# --- Глобальные переменные для инициализации ---
config = None
vc = None
sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}
# Переменные для путей (можно брать из config после инициализации)
weight_root = ""
index_root = ""
weight_uvr5_root = ""
uvr5_names = []
default_batch_size = 1
gpus = ""

def load_model_lists():
    """Загружает списки доступных моделей и индексов."""
    global names, index_paths, uvr5_names, weight_root, index_root, weight_uvr5_root

    weight_root = os.getenv("weight_root", "assets/weights")
    weight_uvr5_root = os.getenv("weight_uvr5_root", "assets/uvr5_weights")
    index_root = os.getenv("index_root", "logs")

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
    if index_root and os.path.exists(index_root):
        try:
            for root, dirs, files in os.walk(index_root, topdown=False):
                for name in files:
                    if name.endswith(".index") and "trained" not in name:
                        index_paths.append(os.path.join(root, name).replace("\\", "/")) # Normalize paths
        except Exception as e:
            logger.warning(f"Could not list indices in {index_root}: {e}")
    else:
        logger.warning(f"Index directory '{index_root}' not found or not set.")

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

def setup_gpu_info():
    """Определяет информацию о GPU и батч-сайз по умолчанию."""
    global ngpu, gpu_infos, mem, if_gpu_ok, gpu_info, default_batch_size, gpus

    ngpu = torch.cuda.device_count()
    gpu_infos = []
    mem = []
    if_gpu_ok = False

    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
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

    gpu_info=""
    default_batch_size = 1
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        default_batch_size = min(mem) // 2 if mem else 1
        gpus = "-".join([i[0] for i in gpu_infos]) if gpu_infos else ""
        logger.info(f"Detected compatible GPUs:\n{gpu_info}")
        logger.info(f"Default batch size set to: {default_batch_size}")
        logger.info(f"Detected GPU IDs: {gpus}")
    else:
        gpu_info = "No suitable NVIDIA GPU found."
        gpus = ""
        logger.warning(gpu_info)

def main():
    global config, vc, default_batch_size, gpus

    # Загружаем списки моделей ДО парсинга аргументов, чтобы использовать их в choices
    load_model_lists()
    setup_gpu_info() # Определяем инфо о GPU и default_batch_size

    # --- Parse CLI arguments ---
    parser = argparse.ArgumentParser(description="RVC CLI Tool", add_help=True)
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., cpu, cuda:0, mps, xpu:0). Overrides auto-detection.")
    parser.add_argument("--half", action='store_true', help="Force half precision (if available). Overrides auto-detection.")
    parser.add_argument("--nohalf", action='store_true', help="Force full precision. Overrides auto-detection.")
    # Дополнительные корневые аргументы можно добавить сюда, если они нужны до инициализации Config

    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True) # required=True

    # --- Preprocess Command ---
    parser_preprocess = subparsers.add_parser('preprocess', help='Preprocess audio dataset')
    parser_preprocess.add_argument('--trainset_dir', type=str, required=True, help='Path to the training dataset directory')
    parser_preprocess.add_argument('--exp_dir', type=str, required=True, help='Experiment directory name (under ./logs)')
    parser_preprocess.add_argument('--sr', type=str, required=True, choices=list(sr_dict.keys()), help='Target sample rate')
    parser_preprocess.add_argument('--n_cpu', type=int, default=None, help='Number of CPU processes (default: auto based on config)')
    # Добавляем .set_defaults с отложенным импортом
    parser_preprocess.set_defaults(func=lambda args, cfg: __import__('rvc_cli.preprocess', fromlist=['preprocess_dataset_cli']).preprocess_dataset_cli(args.trainset_dir, args.exp_dir, args.sr, args.n_cpu if args.n_cpu is not None else int(np.ceil(cfg.n_cpu / 1.5)), cfg))

    # --- Extract Features Command ---
    parser_extract = subparsers.add_parser('extract', help='Extract features (and optionally F0 pitch)')
    parser_extract.add_argument('--exp_dir', type=str, required=True, help='Experiment directory name (must exist under ./logs)')
    parser_extract.add_argument('--gpus', type=str, default=gpus, help='GPU IDs for feature extraction, separated by "-", e.g., "0-1"')
    parser_extract.add_argument('--n_cpu', type=int, default=None, help='Number of CPU processes for F0 extraction (default: auto based on config)')
    parser_extract.add_argument('--f0', action='store_true', help='Enable F0 pitch extraction')
    parser_extract.add_argument('--f0_method', type=str, default='rmvpe_gpu', choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"], help='F0 extraction method')
    parser_extract.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'], help='Model version (affects feature dimension)')
    parser_extract.add_argument('--gpus_rmvpe', type=str, default=None, help='GPU IDs for RMVPE F0 extraction, e.g., "0-0-1" (default: use --gpus argument)')
    parser_extract.set_defaults(func=lambda args, cfg: __import__('rvc_cli.extract', fromlist=['extract_f0_feature_cli']).extract_f0_feature_cli(args.gpus, args.n_cpu if args.n_cpu is not None else int(np.ceil(cfg.n_cpu / 1.5)), args.f0_method, args.f0, args.exp_dir, args.version, args.gpus_rmvpe if args.gpus_rmvpe is not None else args.gpus, cfg))

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
    parser_train.set_defaults(func=lambda args, cfg: __import__('rvc_cli.train', fromlist=['click_train_cli']).click_train_cli(args.exp_dir, args.sr, args.f0, args.spk_id, args.save_epoch, args.total_epoch, args.batch_size if args.batch_size is not None else default_batch_size, args.save_latest, args.pretrained_g, args.pretrained_d, args.gpus, args.cache_gpu, args.save_weights, args.version, cfg))

    # --- Train Index Command ---
    parser_index = subparsers.add_parser('index', help='Train the FAISS index')
    parser_index.add_argument('--exp_dir', type=str, required=True, help='Experiment directory name')
    parser_index.add_argument('--version', type=str, default='v2', choices=['v1', 'v2'], help='Model version (affects feature dimension)')
    parser_index.set_defaults(func=lambda args, cfg: __import__('rvc_cli.index', fromlist=['train_index_cli']).train_index_cli(args.exp_dir, args.version, cfg))

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
    parser_train_all.set_defaults(func=lambda args, cfg: __import__('rvc_cli.train_all', fromlist=['train1key_cli']).train1key_cli(args.exp_dir, args.sr, args.f0, args.trainset_dir, args.spk_id, args.n_cpu if args.n_cpu is not None else int(np.ceil(cfg.n_cpu / 1.5)), args.f0_method, args.save_epoch, args.total_epoch, args.batch_size if args.batch_size is not None else default_batch_size, args.save_latest, args.pretrained_g, args.pretrained_d, args.gpus, args.cache_gpu, args.save_weights, args.version, args.gpus_rmvpe if args.gpus_rmvpe is not None else args.gpus, cfg))

    # --- Infer One Command ---
    parser_infer_one = subparsers.add_parser('infer-one', help='Perform single audio file inference')
    parser_infer_one.add_argument('--sid', type=str, required=True, choices=names, help='Speaker model name from assets/weights')
    parser_infer_one.add_argument('--input_audio', type=str, required=True, help='Path to the input audio file')
    parser_infer_one.add_argument('--output_audio', type=str, required=True, help='Path to save the output audio file')
    parser_infer_one.add_argument('--transpose', type=int, default=0, help='Transpose (pitch shift) in semitones')
    parser_infer_one.add_argument('--f0_method', type=str, default='rmvpe', choices=["pm", "harvest", "crepe", "rmvpe"], help='F0 prediction method')
    parser_infer_one.add_argument('--index_path', type=str, default=None, help='Path to the FAISS index file (e.g., logs/myvoice/added_...index)')
    parser_infer_one.add_argument('--index_rate', type=float, default=0.75, help='Feature retrieval ratio (0 to 1)')
    parser_infer_one.add_argument('--filter_radius', type=int, default=3, help='Median filter radius for Harvest F0 (>=3 applies filter)')
    parser_infer_one.add_argument('--resample_sr', type=int, default=0, help='Resample output audio to this SR (0 for no resampling)')
    parser_infer_one.add_argument('--rms_mix_rate', type=float, default=0.25, help='RMS mix rate (0 to 1)')
    parser_infer_one.add_argument('--protect', type=float, default=0.33, help='Protection for consonants/breaths (0 to 0.5)')
    parser_infer_one.add_argument('--f0_file', type=str, default=None, help='Optional external F0 file path')
    parser_infer_one.set_defaults(func=lambda args, cfg: __import__('rvc_cli.infer', fromlist=['infer_one_cli']).infer_one_cli(args.sid, args.input_audio, args.output_audio, args.transpose, args.f0_method, args.index_path, None, args.index_rate, args.filter_radius, args.resample_sr, args.rms_mix_rate, args.protect, args.f0_file, vc, cfg))

    # --- Infer Batch Command ---
    parser_infer_batch = subparsers.add_parser('infer-batch', help='Perform batch inference on multiple audio files or a directory')
    parser_infer_batch.add_argument('--sid', type=str, required=True, choices=names, help='Speaker model name from assets/weights')
    parser_infer_batch.add_argument('--input_paths', type=str, nargs='+', required=True, help='Paths to input audio files OR a single directory path')
    parser_infer_batch.add_argument('--output_dir', type=str, default='output', help='Directory to save output audio files')
    parser_infer_batch.add_argument('--transpose', type=int, default=0, help='Transpose (pitch shift) in semitones')
    parser_infer_batch.add_argument('--f0_method', type=str, default='rmvpe', choices=["pm", "harvest", "crepe", "rmvpe"], help='F0 prediction method')
    parser_infer_batch.add_argument('--index_path', type=str, default=None, help='Path to the FAISS index file')
    parser_infer_batch.add_argument('--index_rate', type=float, default=1.0, help='Feature retrieval ratio (0 to 1)')
    parser_infer_batch.add_argument('--filter_radius', type=int, default=3, help='Median filter radius for Harvest F0 (>=3 applies filter)')
    parser_infer_batch.add_argument('--resample_sr', type=int, default=0, help='Resample output audio to this SR (0 for no resampling)')
    parser_infer_batch.add_argument('--rms_mix_rate', type=float, default=1.0, help='RMS mix rate (0 to 1)')
    parser_infer_batch.add_argument('--protect', type=float, default=0.33, help='Protection for consonants/breaths (0 to 0.5)')
    parser_infer_batch.add_argument('--format', type=str, default='wav', choices=['wav', 'flac', 'mp3', 'm4a'], help='Output audio format')
    parser_infer_batch.set_defaults(func=lambda args, cfg: __import__('rvc_cli.infer', fromlist=['infer_batch_cli']).infer_batch_cli(args.sid, args.input_paths, args.output_dir, args.transpose, args.f0_method, args.index_path, None, args.index_rate, args.filter_radius, args.resample_sr, args.rms_mix_rate, args.protect, args.format, vc, cfg))

    # --- UVR5 Command ---
    parser_uvr = subparsers.add_parser('uvr', help='Separate vocals and instrumentals using UVR5')
    parser_uvr.add_argument('--model_name', type=str, required=False, choices=uvr5_names + ["list"] if uvr5_names else ["list"], help='UVR5 model name or "list" to show available models.')
    parser_uvr.add_argument('--input_paths', type=str, nargs='+', required=False, help='Paths to input audio files OR a single directory path (required unless listing models)')
    parser_uvr.add_argument('--output_dir_vocals', type=str, default='output/vocals', help='Directory to save vocal tracks')
    parser_uvr.add_argument('--output_dir_instrumental', type=str, default='output/instrumental', help='Directory to save instrumental tracks')
    parser_uvr.add_argument('--agg', type=int, default=10, help='Aggressiveness for vocal extraction (0-20)')
    parser_uvr.add_argument('--format', type=str, default='flac', choices=['wav', 'flac', 'mp3', 'm4a'], help='Output audio format')
    parser_uvr.set_defaults(func=lambda args, cfg: __import__('rvc_cli.uvr', fromlist=['uvr_cli']).uvr_cli(args.model_name, args.input_paths, args.output_dir_vocals, args.output_dir_instrumental, args.agg, args.format, cfg, uvr5_names)) # Передаем список моделей

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
    parser_ckpt_merge.set_defaults(func=lambda args, cfg: __import__('rvc_cli.ckpt_utils', fromlist=['merge_cli']).merge_cli(args.ckpt_a, args.ckpt_b, args.alpha, args.sr, args.f0, args.info, args.save_name, args.version, cfg))

    # --- Ckpt Modify Command ---
    parser_ckpt_modify = subparsers.add_parser('ckpt-modify', help='Modify metadata info string of an RVC model')
    parser_ckpt_modify.add_argument('--ckpt_path', type=str, required=True, help='Path to the model (.pth) to modify')
    parser_ckpt_modify.add_argument('--info', type=str, required=True, help='New info string for model metadata')
    parser_ckpt_modify.add_argument('--save_name', type=str, default="", help='Optional: New filename (without .pth). Overwrites if empty.')
    parser_ckpt_modify.set_defaults(func=lambda args, cfg: __import__('rvc_cli.ckpt_utils', fromlist=['change_info_cli']).change_info_cli(args.ckpt_path, args.info, args.save_name, cfg))

    # --- Ckpt Show Command ---
    parser_ckpt_show = subparsers.add_parser('ckpt-show', help='Show metadata info of an RVC model')
    parser_ckpt_show.add_argument('--ckpt_path', type=str, required=True, help='Path to the model (.pth)')
    parser_ckpt_show.set_defaults(func=lambda args, cfg: __import__('rvc_cli.ckpt_utils', fromlist=['show_info_cli']).show_info_cli(args.ckpt_path, cfg))

    # --- Ckpt Extract Command ---
    parser_ckpt_extract = subparsers.add_parser('ckpt-extract', help='Extract a small inference model from a large training checkpoint')
    parser_ckpt_extract.add_argument('--ckpt_path', type=str, required=True, help='Path to the large training checkpoint (e.g., G_xxxxx.pth)')
    parser_ckpt_extract.add_argument('--save_name', type=str, required=True, help='Filename (without .pth) to save the extracted model (in assets/weights)')
    parser_ckpt_extract.add_argument('--sr', type=str, required=True, choices=list(sr_dict.keys()), help='Target sample rate')
    parser_ckpt_extract.add_argument('--if_f0', type=str, required=True, choices=['0', '1'], help='Whether the model uses F0 pitch (1 for yes, 0 for no)')
    parser_ckpt_extract.add_argument('--info', type=str, default="", help='Optional info string for extracted model metadata')
    parser_ckpt_extract.add_argument('--version', type=str, required=True, choices=['v1', 'v2'], help='Model version')
    parser_ckpt_extract.set_defaults(func=lambda args, cfg: __import__('rvc_cli.ckpt_utils', fromlist=['extract_small_model_cli']).extract_small_model_cli(args.ckpt_path, args.save_name, args.sr, args.if_f0, args.info, args.version, cfg))

    # --- Export ONNX Command ---
    parser_export_onnx = subparsers.add_parser('export-onnx', help='Export an RVC model to ONNX format')
    parser_export_onnx.add_argument('--ckpt_path', type=str, required=True, help='Path to the RVC model (.pth)')
    parser_export_onnx.add_argument('--onnx_path', type=str, required=True, help='Path to save the output ONNX model')
    parser_export_onnx.set_defaults(func=lambda args, cfg: __import__('rvc_cli.onnx_export', fromlist=['export_onnx_cli']).export_onnx_cli(args.ckpt_path, args.onnx_path, cfg))


    # --- Parse arguments and execute function ---
    args = parser.parse_args()

    # Инициализация Config *после* парсинга корневых аргументов
    config = Config() # Config теперь синглтон
    # Переопределение device и is_half
    if args.device is not None:
        logger.info(f"Overriding device with CLI argument: {args.device}")
        config.device = args.device
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

    logger.info(f"Using device: {config.device}")
    logger.info(f"Using half precision: {config.is_half}")

    # Инициализация VC после Config, если команда требует VC
    if args.command in ['infer-one', 'infer-batch']:
        logger.info("Initializing VC...")
        vc = VC(config)
        # Загрузка моделей Hubert произойдет внутри VC при необходимости


    # Обработка команды 'list' для UVR
    if args.command == 'uvr' and args.model_name == 'list':
        print("\nAvailable UVR5 models:")
        if uvr5_names:
            for name in sorted(uvr5_names):
                print(f"- {name}")
        else:
            print("  No models found (check assets/uvr5_weights).")
        print("\nUse the 'uvr' command with '--model_name <model>' and input/output paths to run separation.")
        sys.exit(0)

    # Вызов функции команды
    if hasattr(args, 'func'):
        # Передаем args и config в функцию
        args.func(args, config)
    else:
        # Если команда не была передана (хотя subparsers required=True)
        parser.print_help(sys.stderr)
        sys.exit("Error: No command provided.")


if __name__ == "__main__":
    # Добавлено для предотвращения ошибки рекурсии multiprocessing в Windows/macOS
    if platform.system() == "Windows" or platform.system() == "Darwin":
         multiprocessing.freeze_support() # Нужно вызывать здесь
    main()