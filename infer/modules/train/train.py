# Файл: infer/modules/train/train.py
import os
import sys
import logging

# --- Добавлено: Явное добавление корневой папки проекта в sys.path ---
# Определяем корневую директорию относительно текущего файла
# infer/modules/train/train.py -> infer/modules/train -> infer/modules -> infer -> RVC-KAGGLE (4 уровня вверх)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Добавляем в начало пути поиска
    # print(f"[train.py] Added to sys.path: {project_root}") # Для отладки, можно убрать
# --- Конец добавления ---

# Теперь логгер и импорты можно инициализировать
logger = logging.getLogger(__name__)

import datetime

# Импорт utils теперь должен работать
from infer.lib.train import utils

# --- Остальной код файла train.py (без изменений по сравнению с предыдущей версией) ---
hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        from infer.modules.ipex.gradscaler import gradscaler_init
        from torch.xpu.amp import autocast

        GradScaler = gradscaler_init()
        ipex_init()
    else:
        # Проверяем наличие CUDA перед импортом GradScaler и autocast
        if torch.cuda.is_available():
            from torch.cuda.amp import GradScaler, autocast
        else: # Если CUDA нет, создаем "пустышки"
            logger.warning("CUDA not available, using dummy GradScaler and autocast for CPU/MPS.")
            class GradScaler:
                def __init__(self, enabled=False): self.enabled = enabled
                def scale(self, loss): return loss
                def unscale_(self, optimizer): pass
                def step(self, optimizer): optimizer.step()
                def update(self): pass
            from contextlib import nullcontext
            autocast = nullcontext

except Exception:
    # Проверяем наличие CUDA перед импортом GradScaler и autocast
    if torch.cuda.is_available():
        from torch.cuda.amp import GradScaler, autocast
    else: # Если CUDA нет, создаем "пустышки"
        logger.warning("CUDA not available, using dummy GradScaler and autocast for CPU/MPS.")
        class GradScaler:
            def __init__(self, enabled=False): self.enabled = enabled
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def step(self, optimizer): optimizer.step()
            def update(self): pass
        from contextlib import nullcontext
        autocast = nullcontext


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime # Импортируем time как ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import savee

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    is_cuda_available = torch.cuda.is_available()
    is_mps_available = torch.backends.mps.is_available()
    is_xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()

    if is_cuda_available:
        n_gpus = torch.cuda.device_count()
    elif is_xpu_available:
        n_gpus = torch.xpu.device_count()
    elif is_mps_available:
        n_gpus = 1
    else:
        n_gpus = 0

    if n_gpus == 0:
        print("NO GPU/XPU/MPS DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1 # Запускаем один процесс для CPU

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    # Получаем логгер до запуска процессов, чтобы главный процесс тоже мог его использовать
    # Используем функцию из модуля utils, который теперь импортируется корректно
    logger = utils.get_logger(hps.model_dir)
    logger.info(f"Detected {n_gpus} device(s). Starting training...")

    if n_gpus > 1 and not is_cuda_available and not is_xpu_available:
         logger.warning("Distributed training requires CUDA or XPU. Running on a single process.")
         n_gpus = 1

    # Устанавливаем метод запуска для multiprocessing, если еще не установлен
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method("spawn") # 'spawn' безопаснее для CUDA

    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger), # Передаем логгер
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps, logger: logging.Logger): # Принимаем логгер
    global global_step

    is_main_process = rank == 0
    is_distributed = n_gpus > 1
    device_id = rank # Используем rank как device_id для CUDA/XPU

    if is_distributed:
        # Определяем бэкенд в зависимости от доступности
        if torch.cuda.is_available():
            backend = 'nccl'
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            backend = 'ccl' # Intel CCL
        else:
            backend = 'gloo' # Fallback для CPU/MPS/etc.

        try:
            dist.init_process_group(
                backend=backend, init_method="env://", world_size=n_gpus, rank=rank
            )
        except Exception as e:
            # Используем логгер, переданный как аргумент
            if logger: logger.error(f"Rank {rank}: Failed to initialize process group (backend: {backend}): {e}")
            # Попытка с fallback backend='gloo', если не CPU
            if backend != 'gloo':
                if logger: logger.info(f"Rank {rank}: Trying fallback backend 'gloo'...")
                try:
                    dist.init_process_group(
                        backend='gloo', init_method="env://", world_size=n_gpus, rank=rank
                    )
                except Exception as e_gloo:
                     if logger: logger.error(f"Rank {rank}: Failed to initialize process group with fallback backend 'gloo': {e_gloo}")
                     return # Не можем продолжить без группы процессов

    # Установка устройства после инициализации группы процессов
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        current_device = torch.device(f"cuda:{device_id}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
         torch.xpu.set_device(device_id)
         current_device = torch.device(f"xpu:{device_id}")
    elif torch.backends.mps.is_available() and not is_distributed: # MPS не поддерживает DDP
         current_device = torch.device("mps")
    else:
         current_device = torch.device("cpu")

    if logger: logger.info(f"Rank {rank}/{n_gpus} started on device: {current_device}")

    writer, writer_eval = None, None # Инициализируем None
    if is_main_process:
        # logger уже есть
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir) # Можно раскомментировать при необходимости
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    torch.manual_seed(hps.train.seed + rank) # Добавляем rank к seed для разных процессов

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)

    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * (1 if not is_distributed else n_gpus),
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus if is_distributed else 1,
        rank=rank,
        shuffle=True,
    )

    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False, # Перемешивание управляется сэмплером
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )

    net_g = net_g.to(current_device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    net_d = net_d.to(current_device)

    # Применяем FP16 если нужно
    if hps.train.fp16_run and hasattr(torch, "xpu") and torch.xpu.is_available():
        net_g, optim_g = ipex.optimize(net_g, dtype=torch.bfloat16, optimizer=optim_g)
        net_d, optim_d = ipex.optimize(net_d, dtype=torch.bfloat16, optimizer=optim_d)
        if logger: logger.info("Using IPEX BF16 optimization.")
    elif hps.train.fp16_run:
         if logger: logger.info("Using FP16 training.")
         # GradScaler уже инициализирован выше с учетом CUDA

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if is_distributed:
        # Оборачиваем в DDP
        if torch.cuda.is_available():
            # find_unused_parameters может быть нужен, если часть модели не используется
            net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
            net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
             net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
             net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
        else: # Gloo для CPU/MPS
             net_g = DDP(net_g, find_unused_parameters=True)
             net_d = DDP(net_d, find_unused_parameters=True)

    try:
        # Загрузка чекпоинтов D и G
        latest_d_path = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        if latest_d_path:
             _, _, _, epoch_str = utils.load_checkpoint(latest_d_path, net_d, optim_d)
             if is_main_process: logger.info("Loaded Discriminator checkpoint.")
        else:
            epoch_str = 1 # Начинаем с 1 эпохи, если чекпоинта D нет

        latest_g_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        if latest_g_path:
            # Перезаписываем epoch_str, если G новее D (хотя обычно они сохраняются вместе)
            _, _, _, epoch_str = utils.load_checkpoint(latest_g_path, net_g, optim_g)
            if is_main_process: logger.info("Loaded Generator checkpoint.")
        elif not latest_d_path: # Если нет ни D, ни G чекпоинтов
             epoch_str = 1

        # Рассчитываем global_step
        # Учитываем, что len(train_loader) вернет размер батча для *одного* процесса
        # В распределенном режиме реальное количество шагов = len(train_loader)
        steps_per_epoch = len(train_loader)
        global_step = (epoch_str - 1) * steps_per_epoch
        if is_main_process: logger.info(f"Resuming from epoch {epoch_str}, global step {global_step}")

    except Exception as e:
        if logger: logger.warning(f"Could not load checkpoints, starting from scratch or pretrain: {e}")
        epoch_str = 1
        global_step = 0
        # Логика загрузки претрейнов (остается без изменений)
        if hps.pretrainG != "":
            if is_main_process: logger.info("Loading pretrained Generator: %s" % (hps.pretrainG))
            model_dict = torch.load(hps.pretrainG, map_location="cpu")["model"]
            model_to_load = net_g.module if hasattr(net_g, "module") else net_g
            logger.info(model_to_load.load_state_dict(model_dict, strict=False))
        if hps.pretrainD != "":
            if is_main_process: logger.info("Loading pretrained Discriminator: %s" % (hps.pretrainD))
            model_dict = torch.load(hps.pretrainD, map_location="cpu")["model"]
            model_to_load = net_d.module if hasattr(net_d, "module") else net_d
            logger.info(model_to_load.load_state_dict(model_dict, strict=False))


    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    # GradScaler инициализируется один раз вне зависимости от CUDA/XPU/CPU
    # enabled=False если FP16 не используется
    scaler = GradScaler(enabled=hps.train.fp16_run and (torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())))


    cache = [] # Кэш теперь локальный для каждого процесса
    if hps.if_cache_data_in_gpu:
         if logger: logger.info(f"Rank {rank}: Caching method selected, pre-caching data...")
         # Вызываем train_and_evaluate один раз с флагом для кэширования
         train_and_evaluate(
             rank, 0, hps, [net_g, net_d], [optim_g, optim_d],
             [scheduler_g, scheduler_d], scaler, [train_loader, None],
             logger if is_main_process else None,
             [writer, writer_eval] if is_main_process else None,
             cache, only_cache=True
         )
         if logger: logger.info(f"Rank {rank}: Caching finished.")
         # Сбрасываем global_step, так как реальное обучение еще не началось
         global_step = (epoch_str - 1) * steps_per_epoch

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # Вызываем train_and_evaluate для всех процессов
        train_and_evaluate(
            rank,
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            scaler,
            [train_loader, None],
            logger if is_main_process else None, # Логгер только для rank 0
            [writer, writer_eval] if is_main_process else None, # Writers только для rank 0
            cache,
        )
        scheduler_g.step()
        scheduler_d.step()

    # Завершение для всех процессов
    if is_distributed:
        dist.destroy_process_group()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache, only_cache=False
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    # scheduler_g, scheduler_d = schedulers # Не используется напрямую в цикле
    train_loader, eval_loader = loaders
    writer, writer_eval = (None, None) # Инициализация по умолчанию
    if writers is not None:
        writer, writer_eval = writers # Распаковка, если writers передан

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    # Определяем устройство (дублируется из run, но для ясности)
    if torch.cuda.is_available():
        current_device = torch.device(f"cuda:{rank}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
         current_device = torch.device(f"xpu:{rank}")
    elif torch.backends.mps.is_available():
         current_device = torch.device("mps")
    else:
         current_device = torch.device("cpu")

    # --- Кэширование данных (если only_cache=True) ---
    if only_cache:
        if hps.if_cache_data_in_gpu and not cache: # Кэшируем только если кэш пуст
            # Логгер доступен только для rank 0
            if logger: logger.info(f"Rank {rank}: Starting data caching...")
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0 == 1: (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid) = info
                else: (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid) = info
                # Load on Device
                phone, phone_lengths = phone.to(current_device, non_blocking=True), phone_lengths.to(current_device, non_blocking=True)
                if hps.if_f0 == 1: pitch, pitchf = pitch.to(current_device, non_blocking=True), pitchf.to(current_device, non_blocking=True)
                sid, spec, spec_lengths, wave = sid.to(current_device, non_blocking=True), spec.to(current_device, non_blocking=True), spec_lengths.to(current_device, non_blocking=True), wave.to(current_device, non_blocking=True)
                # Cache on list
                if hps.if_f0 == 1: cache.append((batch_idx, (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid)))
                else: cache.append((batch_idx, (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid)))
            if logger: logger.info(f"Rank {rank}: Caching complete. {len(cache)} batches cached.")
        return # Завершаем функцию, если только кэшировали

    # --- Основной цикл обучения ---
    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu:
        data_iterator = cache
        if epoch > 1: # Перемешиваем кэш для каждой эпохи, кроме первой (уже перемешан в run)
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    steps_in_epoch = len(train_loader)
    epoch_loss_disc = 0.0
    epoch_loss_gen = 0.0
    epoch_loss_fm = 0.0
    epoch_loss_mel = 0.0
    epoch_loss_kl = 0.0
    epoch_loss_gen_all = 0.0
    epoch_start_time = ttime()

    # Итерируемся по данным
    for batch_idx_enum, info_packed in data_iterator:
        # Распаковываем индекс и данные в зависимости от источника (кэш или лоадер)
        if hps.if_cache_data_in_gpu:
            original_batch_idx, info = info_packed
            # Используем batch_idx_enum для отслеживания прогресса внутри эпохи
            current_batch_idx = batch_idx_enum
        else:
            current_batch_idx, info = batch_idx_enum # Используем batch_idx из enumerate

        step_start_time = ttime()
        # Распаковка данных
        if hps.if_f0 == 1:
            (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid) = info # wave_lengths не используется
        else:
            (phone, phone_lengths, spec, spec_lengths, wave, _, sid) = info # wave_lengths не используется

        # Перенос на устройство, если не из кэша
        if not hps.if_cache_data_in_gpu:
            phone, phone_lengths = phone.to(current_device, non_blocking=True), phone_lengths.to(current_device, non_blocking=True)
            if hps.if_f0 == 1: pitch, pitchf = pitch.to(current_device, non_blocking=True), pitchf.to(current_device, non_blocking=True)
            sid, spec, spec_lengths, wave = sid.to(current_device, non_blocking=True), spec.to(current_device, non_blocking=True), spec_lengths.to(current_device, non_blocking=True), wave.to(current_device, non_blocking=True)

        # --- Вычисления и обратное распространение ---
        # (Здесь код остается без изменений, как в предыдущем ответе, с autocast и scaler)
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
            if hps.train.fp16_run: y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)

            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        # --- Конец вычислений ---

        step_end_time = ttime()
        step_duration = step_end_time - step_start_time

        if rank == 0: # Логируем только в rank 0
            epoch_loss_disc += loss_disc.item()
            epoch_loss_gen += loss_gen.item()
            epoch_loss_fm += loss_fm.item()
            epoch_loss_mel += loss_mel.item()
            epoch_loss_kl += loss_kl.item()
            epoch_loss_gen_all += loss_gen_all.item()

            log_frequency = min(getattr(hps.train, 'log_interval', 100), 10)
            if global_step % log_frequency == 0:
                lr = optim_g.param_groups[0]["lr"]
                loss_mel_display = min(loss_mel.item(), 75) if isinstance(loss_mel, torch.Tensor) else loss_mel
                loss_kl_display = min(loss_kl.item(), 9) if isinstance(loss_kl, torch.Tensor) else loss_kl
                steps_done = current_batch_idx + 1
                steps_left = steps_in_epoch - steps_done
                avg_step_time = (step_end_time - epoch_start_time) / steps_done if steps_done > 0 else step_duration
                eta_seconds = steps_left * avg_step_time
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds))) if eta_seconds >= 0 else "N/A"

                logger.info(
                    f"Epoch: {epoch} [{steps_done}/{steps_in_epoch} ({100. * steps_done / steps_in_epoch:.0f}%)] | "
                    f"Step: {global_step} | LR: {lr:.2e} | Grad Norms: [D:{grad_norm_d or 0:.2f}, G:{grad_norm_g or 0:.2f}] | "
                    f"Losses: [D:{loss_disc.item():.3f}, G:{loss_gen.item():.3f}, FM:{loss_fm.item():.3f}, MEL:{loss_mel_display:.3f}, KL:{loss_kl_display:.3f}] | "
                    f"Step Time: {step_duration:.2f}s | Epoch ETA: {eta_str}"
                )

            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d or 0,
                    "grad_norm_g": grad_norm_g or 0,
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/kl": loss_kl,
                    "step_time": step_duration
                }
                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
    # /Конец цикла по батчам

    # Логирование средних потерь по завершении эпохи (только в rank 0)
    if rank == 0:
        avg_loss_disc = epoch_loss_disc / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_loss_gen = epoch_loss_gen / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_loss_fm = epoch_loss_fm / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_loss_mel = epoch_loss_mel / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_loss_kl = epoch_loss_kl / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_loss_gen_all = epoch_loss_gen_all / steps_in_epoch if steps_in_epoch > 0 else 0
        epoch_duration = ttime() - epoch_start_time

        logger.info(f"Epoch {epoch} finished. {epoch_recorder.record()} | Duration: {datetime.timedelta(seconds=int(epoch_duration))}")
        logger.info(
            f"  Avg Losses: [Disc: {avg_loss_disc:.3f}, Gen_All: {avg_loss_gen_all:.3f} "
            f"(G:{avg_loss_gen:.3f}, FM:{avg_loss_fm:.3f}, MEL:{avg_loss_mel:.3f}, KL:{avg_loss_kl:.3f})]"
        )
        # Запись средних потерь в TensorBoard
        if writer is not None:
            writer.add_scalar("epoch_avg_loss/d/total", avg_loss_disc, epoch)
            writer.add_scalar("epoch_avg_loss/g/total", avg_loss_gen_all, epoch)
            writer.add_scalar("epoch_avg_loss/g/gen", avg_loss_gen, epoch)
            writer.add_scalar("epoch_avg_loss/g/fm", avg_loss_fm, epoch)
            writer.add_scalar("epoch_avg_loss/g/mel", avg_loss_mel, epoch)
            writer.add_scalar("epoch_avg_loss/g/kl", avg_loss_kl, epoch)
            writer.add_scalar("epoch_duration_seconds", epoch_duration, epoch)

    # Сохранение чекпоинтов (только в rank 0)
    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step))
            )
            utils.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step))
            )
        else:
            utils.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(2333333))
            )
            utils.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(2333333))
            )
        if rank == 0 and getattr(hps, 'save_every_weights', "0") == "1": # Проверяем наличие атрибута
            model_to_save = net_g.module if hasattr(net_g, "module") else net_g
            ckpt = model_to_save.state_dict()
            logger.info(
                "Saving small ckpt %s_e%s:%s" % (
                    hps.name, epoch,
                    savee(ckpt, hps.sample_rate, hps.if_f0,
                          hps.name + "_e%s_s%s" % (epoch, global_step),
                          epoch, hps.version, hps)
                )
            )

    # Завершение обучения (только в rank 0)
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")
        model_to_save = net_g.module if hasattr(net_g, "module") else net_g
        ckpt = model_to_save.state_dict()
        logger.info(
            "Saving final ckpt:%s" % (
                savee(ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps)
            )
        )
        sleep(1)
        # Не используем os._exit в распределенном режиме, позволяем процессам завершиться штатно


if __name__ == "__main__":
    # Установка метода запуска 'spawn' рекомендуется для CUDA
    if mp.get_start_method(allow_none=True) != 'spawn':
         mp.set_start_method("spawn", force=True)
    main()