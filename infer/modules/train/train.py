import os
import sys
import logging

# Этот логгер logger = logging.getLogger(__name__) на верхнем уровне модуля
# будет использовать конфигурацию корневого логгера, установленную в run_rvc_cli.py.
# Его сообщения (если вы будете его использовать здесь) пойдут в консоль с уровнем INFO.
# Для основного логирования обучения мы будем использовать логгер, создаваемый в rank 0.
# logger = logging.getLogger(__name__) # Можно оставить, если используется для чего-то еще на уровне модуля

now_dir = os.getcwd()
if now_dir not in sys.path: # Добавляем now_dir, если его еще нет
    sys.path.append(now_dir)

import datetime

from infer.lib.train import utils # utils содержит get_logger

# hps получается в main_ddp_entry_point, а не глобально здесь
# hps = utils.get_hparams() # УБРАНО ОТСЮДА
# os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",") # ПЕРЕМЕЩЕНО В main_ddp_entry_point
# n_gpus = len(hps.gpus.split("-")) # ПЕРЕМЕЩЕНО В main_ddp_entry_point

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
        from torch.cuda.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False # Обычно True для ускорения, если размеры входа не меняются
from time import sleep
from time import time as ttime

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

# Динамический импорт моделей в функции run, после получения hps
# if hps.version == "v1": ... # ПЕРЕМЕЩЕНО В run

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


def run(rank, world_size, hps_config): # logger убран из аргументов, hps переименован для ясности
    global global_step

    # Инициализация логгера и TensorBoard writers только в процессе rank 0
    current_process_logger = None
    writer = None
    writer_eval = None

    if rank == 0:
        # Получаем (или создаем) логгер непосредственно в этом процессе
        current_process_logger = utils.get_logger(hps_config.model_dir)
        
        # --- НАЧАЛО ИСПРАВЛЕНИЯ ДЛЯ КОНСОЛЬНОГО ВЫВОДА ---
        # Удаляем существующие StreamHandler'ы с stdout у ЭТОГО логгера, чтобы избежать дублей,
        # если этот код будет вызван несколько раз или если utils.get_logger теоретически мог бы его добавить.
        for handler in list(current_process_logger.handlers): # Итерируемся по копии списка
            if isinstance(handler, logging.StreamHandler) and \
               hasattr(handler.stream, 'fileno') and \
               handler.stream.fileno() == sys.stdout.fileno():
                current_process_logger.removeHandler(handler)
                # Используем print в stderr для отладки самого логгера
                print(f"Rank 0 DEBUG: Removed existing stdout StreamHandler from logger '{current_process_logger.name}' to avoid duplicates.", file=sys.stderr)

        # Создаем и добавляем новый StreamHandler для вывода в консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO) # Устанавливаем уровень INFO для консоли

        # Используем формат, аналогичный тому, что в run_rvc_cli.py для единообразия
        cli_formatter_str = '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        cli_formatter = logging.Formatter(cli_formatter_str)
        console_handler.setFormatter(cli_formatter)
        
        current_process_logger.addHandler(console_handler)

        # ВАЖНО: Отключаем дальнейшую пропагацию сообщений от current_process_logger к корневому логгеру.
        current_process_logger.propagate = False

        print(f"Rank 0 DEBUG: Added INFO StreamHandler to logger '{current_process_logger.name}'. Propagation set to {current_process_logger.propagate}.", file=sys.stderr)
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ДЛЯ КОНСОЛЬНОГО ВЫВОДА ---

        current_process_logger.info(f"Rank 0: Logger '{current_process_logger.name}' initialized with FileHandler and new ConsoleHandler.")
        current_process_logger.info(hps_config)
        
        # utils.check_git_hash(hps_config.model_dir) # Раскомментируйте, если используется
        
        writer = SummaryWriter(log_dir=hps_config.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps_config.model_dir, "eval"))
        current_process_logger.info("TensorBoard SummaryWriter initialized.")

    # Настройка DDP группы
    dist.init_process_group(
        backend="gloo", # или "nccl" для NVIDIA GPU, если доступно и предпочтительно
        init_method="env://", # MASTER_ADDR и MASTER_PORT должны быть установлены
        world_size=world_size, 
        rank=rank
    )

    torch.manual_seed(hps_config.train.seed + rank) # Добавляем rank к seed для возможной разной инициализации данных/аугментаций
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank) # Устанавливаем устройство для текущего процесса

    # Динамический импорт моделей на основе hps_config.version
    if hps_config.version == "v1":
        from infer.lib.infer_pack.models import MultiPeriodDiscriminator
        from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
        from infer.lib.infer_pack.models import (
            SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
        )
    else: # Предполагаем v2 или новее
        from infer.lib.infer_pack.models import (
            SynthesizerTrnMs768NSFsid as RVC_Model_f0,
            SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
        )

    if hps_config.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps_config.data.training_files, hps_config.data)
    else:
        train_dataset = TextAudioLoader(hps_config.data.training_files, hps_config.data)
    
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps_config.train.batch_size * world_size, # Общий batch_size на все GPU
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    if hps_config.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    
    train_loader = DataLoader(
        train_dataset,
        num_workers=4, # Можно сделать параметром hps_config.train.num_workers
        shuffle=False, # Shuffle обеспечивается DistributedBucketSampler
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True if int(getattr(hps_config.train, "num_workers", 4)) > 0 else False, # persistent_workers=True, # Если num_workers > 0
        prefetch_factor=int(getattr(hps_config.train, "prefetch_factor", 8)), # prefetch_factor=8, # Можно сделать параметром
    )

    if hps_config.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps_config.data.filter_length // 2 + 1,
            hps_config.train.segment_size // hps_config.data.hop_length,
            **hps_config.model,
            is_half=hps_config.train.fp16_run,
            sr=hps_config.sample_rate, # Передаем sample_rate из hps_config
        )
    else:
        net_g = RVC_Model_nof0(
            hps_config.data.filter_length // 2 + 1,
            hps_config.train.segment_size // hps_config.data.hop_length,
            **hps_config.model,
            is_half=hps_config.train.fp16_run,
        )
    
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    
    net_d = MultiPeriodDiscriminator(hps_config.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps_config.train.learning_rate,
        betas=hps_config.train.betas,
        eps=hps_config.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps_config.train.learning_rate,
        betas=hps_config.train.betas,
        eps=hps_config.train.eps,
    )

    if hasattr(torch, "xpu") and torch.xpu.is_available(): # Ipex / XPU
        pass 
    elif torch.cuda.is_available(): # CUDA
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True) # Добавлено find_unused_parameters
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True) # Добавлено find_unused_parameters
    else: # CPU
        net_g = DDP(net_g, find_unused_parameters=True) # Добавлено find_unused_parameters
        net_d = DDP(net_d, find_unused_parameters=True) # Добавлено find_unused_parameters
    
    epoch_str = 1 
    # global_step уже 0 по умолчанию

    try:
        _, _, _, epoch_str_d = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps_config.model_dir, "D_*.pth"), net_d, optim_d
        )
        if rank == 0 and current_process_logger:
            current_process_logger.info(f"Loaded Discriminator checkpoint, starting from epoch {epoch_str_d}")
        
        _, _, _, epoch_str_g = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps_config.model_dir, "G_*.pth"), net_g, optim_g
        )
        if rank == 0 and current_process_logger:
            current_process_logger.info(f"Loaded Generator checkpoint, starting from epoch {epoch_str_g}")
        
        epoch_str = epoch_str_g 
        global_step = (epoch_str - 1) * len(train_loader) 

    except Exception as e:
        if rank == 0 and current_process_logger:
            current_process_logger.warning(f"Could not load G/D checkpoints, attempting to load pretrained models. Error: {e}")
        epoch_str = 1
        global_step = 0
        if hasattr(hps_config, 'pretrainG') and hps_config.pretrainG: 
            if rank == 0 and current_process_logger:
                current_process_logger.info(f"Loading pretrained Generator: {hps_config.pretrainG}")
            model_g_to_load = net_g.module if hasattr(net_g, "module") else net_g
            try:
                model_g_to_load.load_state_dict(torch.load(hps_config.pretrainG, map_location="cpu")["model"])
            except Exception as load_e:
                if rank == 0 and current_process_logger:
                    current_process_logger.error(f"Failed to load pretrained G: {load_e}")

        if hasattr(hps_config, 'pretrainD') and hps_config.pretrainD: 
            if rank == 0 and current_process_logger:
                current_process_logger.info(f"Loading pretrained Discriminator: {hps_config.pretrainD}")
            model_d_to_load = net_d.module if hasattr(net_d, "module") else net_d
            try:
                model_d_to_load.load_state_dict(torch.load(hps_config.pretrainD, map_location="cpu")["model"])
            except Exception as load_e:
                if rank == 0 and current_process_logger:
                    current_process_logger.error(f"Failed to load pretrained D: {load_e}")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps_config.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps_config.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps_config.train.fp16_run)
    cache = []

    if rank == 0 and current_process_logger:
        current_process_logger.info(f"Starting training from epoch: {epoch_str}")

    for epoch in range(epoch_str, hps_config.train.epochs + 1):
        if rank == 0: 
            train_and_evaluate(
                rank, epoch, hps_config, [net_g, net_d], [optim_g, optim_d],
                [scheduler_g, scheduler_d], scaler, [train_loader, None], 
                current_process_logger, [writer, writer_eval], cache,
            )
        else: 
            train_and_evaluate(
                rank, epoch, hps_config, [net_g, net_d], [optim_g, optim_d],
                [scheduler_g, scheduler_d], scaler, [train_loader, None],
                None, None, cache,
            )
        scheduler_g.step()
        scheduler_d.step()

    if rank == 0 and current_process_logger:
        current_process_logger.info("Training finished.")
    
    dist.destroy_process_group()

def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, current_process_logger, writers, cache
): 
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders 
    
    writer = None
    if writers is not None: 
        writer, _ = writers 

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    data_iterator = None
    if hps.if_cache_data_in_gpu: 
        data_iterator = cache
        if not cache: 
            if rank == 0 and current_process_logger:
                current_process_logger.debug(f"Epoch {epoch}: Building cache for GPU data.")
            for batch_idx_cache, info_cache in enumerate(train_loader): # Изменено имя переменной
                if hps.if_f0 == 1:
                    (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid) = info_cache
                else:
                    (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid) = info_cache
                
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    if hasattr(wave_lengths, 'cuda'): # Добавлена проверка, т.к. wave_lengths может быть не тензором
                         wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                
                if hps.if_f0 == 1:
                    cache.append((batch_idx_cache, (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid)))
                else:
                    cache.append((batch_idx_cache, (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid)))
            if rank == 0 and current_process_logger:
                current_process_logger.debug(f"Epoch {epoch}: Cache built with {len(cache)} batches.")
        
        if cache: 
             shuffle(cache) 
        else: 
            if rank == 0 and current_process_logger:
                current_process_logger.warning(f"Epoch {epoch}: Data cache is empty. Training might not proceed if dataset is also empty.")
            data_iterator = enumerate([]) 

    else: 
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    
    for batch_idx, batch_data_loop in data_iterator: # batch_data_loop - новое имя
        info_loop = None # Инициализация info_loop
        if hps.if_cache_data_in_gpu and cache : 
            original_batch_idx, info_loop = batch_data_loop 
        else: 
            info_loop = batch_data_loop 

        if hps.if_f0 == 1:
            (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid) = info_loop
        else:
            (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid) = info_loop
        
        if not hps.if_cache_data_in_gpu and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            
            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False): 
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax,
                )
            if hps.train.fp16_run: 
                y_hat_mel = y_hat_mel.half()
            
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

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

        if rank == 0 and current_process_logger is not None: 
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                
                # Определение знаменателя для progress_percentage
                denominator = len(train_loader)
                if hps.if_cache_data_in_gpu and cache:
                    denominator = len(cache)
                if denominator == 0: # Избегаем деления на ноль, если train_loader/cache пусты
                    progress_percentage = 0.0
                else:
                    progress_percentage = 100.0 * (batch_idx + 1) / denominator


                current_process_logger.info(
                    "Train Epoch: {} [{:.0f}%] GlobalStep: {}".format(
                        epoch, progress_percentage, global_step
                    )
                )
                
                loss_mel_item = loss_mel.item() if hasattr(loss_mel, 'item') else loss_mel
                loss_kl_item = loss_kl.item() if hasattr(loss_kl, 'item') else loss_kl
                
                if loss_mel_item > 75: loss_mel_item = 75 
                if loss_kl_item > 9: loss_kl_item = 9   

                current_process_logger.info(f"LR: {lr}, Step: {global_step}")
                current_process_logger.info(
                    f"Losses: Disc={loss_disc.item():.3f}, Gen={loss_gen.item():.3f}, "
                    f"FM={loss_fm.item():.3f}, Mel={loss_mel_item:.3f}, KL={loss_kl_item:.3f}"
                )
                
                if writer is not None: 
                    scalar_dict = {
                        "loss/g/total": loss_gen_all.item() if hasattr(loss_gen_all, 'item') else loss_gen_all, # Добавлено .item()
                        "loss/d/total": loss_disc.item() if hasattr(loss_disc, 'item') else loss_disc, # Добавлено .item()
                        "learning_rate": lr, 
                        "grad_norm_d": grad_norm_d if grad_norm_d is not None else 0.0, # Обработка None
                        "grad_norm_g": grad_norm_g if grad_norm_g is not None else 0.0, # Обработка None
                        "loss/g/fm": loss_fm.item() if hasattr(loss_fm, 'item') else loss_fm, # Добавлено .item()
                        "loss/g/mel": loss_mel_item, 
                        "loss/g/kl": loss_kl_item
                    }
                    scalar_dict.update({"loss/g/{}".format(i): v.item() if hasattr(v, 'item') else v for i, v in enumerate(losses_gen)}) # Добавлено .item()
                    scalar_dict.update({"loss/d_r/{}".format(i): v.item() if hasattr(v, 'item') else v for i, v in enumerate(losses_disc_r)})# Добавлено .item()
                    scalar_dict.update({"loss/d_g/{}".format(i): v.item() if hasattr(v, 'item') else v for i, v in enumerate(losses_disc_g)})# Добавлено .item()
                    
                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                        "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    }
                    utils.summarize(
                        writer=writer, global_step=global_step,
                        images=image_dict, scalars=scalar_dict
                    )
        global_step += 1

    if rank == 0 and current_process_logger is not None:
        if epoch % hps.save_every_epoch == 0:
            # Проверяем, является ли hps.if_latest строкой или числом
            if_latest_val = hps.if_latest
            if isinstance(hps.if_latest, str):
                 if_latest_val = int(hps.if_latest)

            if if_latest_val == 0:
                g_path = os.path.join(hps.model_dir, f"G_{global_step}.pth")
                d_path = os.path.join(hps.model_dir, f"D_{global_step}.pth")
            else: 
                g_path = os.path.join(hps.model_dir, "G_2333333.pth")
                d_path = os.path.join(hps.model_dir, "D_2333333.pth")
            
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_path)
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_path)

            if hasattr(hps, 'save_every_weights') and hps.save_every_weights == "1": 
                model_to_save = net_g.module if hasattr(net_g, "module") else net_g
                ckpt_state = model_to_save.state_dict()
                save_name = hps.name + "_e%s_s%s" % (epoch, global_step)
                
                saved_path = savee( 
                    ckpt_state, hps.sample_rate, hps.if_f0,
                    save_name, epoch, hps.version, hps
                )
                current_process_logger.info(f"Saved small checkpoint to: {saved_path}")

        current_process_logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
        
        total_epoch_val = hps.total_epoch
        if isinstance(hps.total_epoch, str):
            total_epoch_val = int(hps.total_epoch)

        if epoch >= total_epoch_val: 
            current_process_logger.info("Target total_epoch reached. Training is done.")
            final_model_to_save = net_g.module if hasattr(net_g, "module") else net_g
            final_ckpt_state = final_model_to_save.state_dict()
            final_save_name = hps.name 
            
            final_saved_path = savee(
                final_ckpt_state, hps.sample_rate, hps.if_f0,
                final_save_name, epoch, hps.version, hps
            )
            current_process_logger.info(f"Saving final small checkpoint: {final_saved_path}")


def main_ddp_entry_point(hps_params_from_cli):
    if hasattr(hps_params_from_cli, "gpus") and hps_params_from_cli.gpus.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = hps_params_from_cli.gpus.replace("-", ",")
        n_gpus_val = len(hps_params_from_cli.gpus.split("-"))
    else: 
        if torch.cuda.is_available():
            n_gpus_val = torch.cuda.device_count()
            if n_gpus_val > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(n_gpus_val)))
                if not hasattr(hps_params_from_cli, "gpus") or not hps_params_from_cli.gpus.strip():
                     hps_params_from_cli.gpus = "-".join(map(str, range(n_gpus_val)))
            else: 
                n_gpus_val = 1 
                os.environ["CUDA_VISIBLE_DEVICES"] = "" 
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): 
             n_gpus_val = 1
             os.environ["CUDA_VISIBLE_DEVICES"] = "" 
             print("INFO: MPS backend detected. Using 1 logical GPU for DDP.")
        elif hasattr(torch, "xpu") and torch.xpu.is_available(): 
             n_gpus_val = 1 
             os.environ["CUDA_VISIBLE_DEVICES"] = "" 
             print("INFO: XPU backend detected. Using 1 logical GPU for DDP (revise if multi-XPU).")
        else: 
            n_gpus_val = 1
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("INFO: No CUDA/MPS/XPU detected. Using CPU with 1 logical GPU for DDP.")
            
    if n_gpus_val < 1:
        print("ERROR: Number of GPUs is less than 1. Cannot start DDP training.")
        return

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    try:
        os.environ["MASTER_PORT"] = str(utils.find_free_port())
    except AttributeError: 
        os.environ["MASTER_PORT"] = "8665" 
        # Выводим предупреждение только если n_gpus_val > 0, чтобы не спамить для CPU
        # Однако, если n_gpus_val=1 для CPU, find_free_port всё равно может быть полезен для избежания конфликтов при нескольких запусках
        print(f"WARNING: utils.find_free_port() not found or failed. Using fixed MASTER_PORT={os.environ['MASTER_PORT']}. This might cause issues if port is busy.", file=sys.stderr)

    children = []
    for i in range(n_gpus_val):
        subproc = mp.Process(target=run, args=(i, n_gpus_val, hps_params_from_cli))
        children.append(subproc)
        subproc.start()

    successful_completion = True
    for i in range(n_gpus_val):
        children[i].join()
        if children[i].exitcode != 0:
            print(f"ERROR: Process for rank {i} exited with code {children[i].exitcode}", file=sys.stderr)
            successful_completion = False
    
    if successful_completion:
        print("All DDP training processes completed successfully.")
    else:
        print("One or more DDP training processes failed.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True) 
    except RuntimeError as e:
        if "context has already been set" not in str(e).lower():
            print(f"Error setting multiprocessing start method: {e}", file=sys.stderr)
            sys.exit(1)

    print("INFO: Running DDP training script directly (__main__).")
    try:
        # Эта строка должна быть адаптирована в зависимости от того, как get_hparams получает аргументы
        # Если get_hparams не использует sys.argv напрямую, а требует передачи аргументов,
        # то здесь нужно будет либо захардкодить путь к конфигу, либо передать аргументы командной строки.
        # Пример: hps_main = utils.get_hparams_from_file("путь/к/вашему/config.json")
        # Или если get_hparams все же работает с sys.argv:
        hps_main = utils.get_hparams() # Убедитесь, что get_hparams может быть вызван так при прямом запуске
    except Exception as e:
        print(f"CRITICAL: Failed to get hparams when running script directly: {e}", file=sys.stderr)
        print("Ensure utils.get_hparams() can correctly parse arguments or load a config when this script is run directly.", file=sys.stderr)
        sys.exit(1)

    main_ddp_entry_point(hps_main)
