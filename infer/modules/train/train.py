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
        current_process_logger.info("Starting training process, rank 0 initializing logger.")
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
        # hps_config.train.batch_size, # Если batch_size в конфиге это per-GPU
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
        persistent_workers=True, # Если num_workers > 0
        prefetch_factor=8, # Можно сделать параметром
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
        # Для XPU DDP может требовать специальной обертки или настроек ipex
        # net_g = DDP(net_g) # Упрощенный вариант, проверить документацию ipex
        # net_d = DDP(net_d)
        pass # Предполагаем, что ipex.optimize уже может обработать это или нужна спец. обертка
    elif torch.cuda.is_available(): # CUDA
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else: # CPU
        net_g = DDP(net_g)
        net_d = DDP(net_d)
    
    epoch_str = 1 # Инициализация по умолчанию
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
        
        # Обычно эпоха должна быть одинаковой для G и D, берем от G или проверяем совпадение
        epoch_str = epoch_str_g 
        global_step = (epoch_str - 1) * len(train_loader) # Вычисляем global_step

    except Exception as e:
        if rank == 0 and current_process_logger:
            current_process_logger.warning(f"Could not load G/D checkpoints, attempting to load pretrained models. Error: {e}")
        epoch_str = 1
        global_step = 0
        # Логика загрузки pretrained моделей
        if hps_config.pretrainG: # Проверяем, что строка не пустая
            if rank == 0 and current_process_logger:
                current_process_logger.info(f"Loading pretrained Generator: {hps_config.pretrainG}")
            # Осторожно с DDP: загрузка state_dict для net_g.module, если net_g уже обернут в DDP
            model_g_to_load = net_g.module if hasattr(net_g, "module") else net_g
            try:
                model_g_to_load.load_state_dict(torch.load(hps_config.pretrainG, map_location="cpu")["model"])
            except Exception as load_e:
                if rank == 0 and current_process_logger:
                    current_process_logger.error(f"Failed to load pretrained G: {load_e}")


        if hps_config.pretrainD: # Проверяем, что строка не пустая
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
        if rank == 0: # Только rank 0 передает логгер и writers
            train_and_evaluate(
                rank,
                epoch,
                hps_config,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None], # eval_loader is None
                current_process_logger, # Передаем логгер, созданный в этом процессе
                [writer, writer_eval], # Передаем writers
                cache,
            )
        else: # Остальные процессы не логируют и не используют writers
            train_and_evaluate(
                rank,
                epoch,
                hps_config,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None, # Logger is None
                None, # Writers is None
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()

    if rank == 0 and current_process_logger:
        current_process_logger.info("Training finished.")
    
    dist.destroy_process_group() # Не забываем очистить группу DDP

def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, current_process_logger, writers, cache
): # logger переименован для ясности
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders # eval_loader здесь None
    
    # writers распаковываются только если они не None (т.е. rank == 0)
    writer = None
    # writer_eval = None # writer_eval не используется для записи в этом цикле
    if writers is not None: # Это будет True только для rank == 0
        writer, _ = writers # _ используется для writer_eval, который не используется здесь

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    data_iterator = None
    if hps.if_cache_data_in_gpu: # Убрано == True для краткости
        data_iterator = cache
        if not cache: # Проверка, что cache не пустой
            if rank == 0 and current_process_logger:
                current_process_logger.debug(f"Epoch {epoch}: Building cache for GPU data.")
            for batch_idx, info in enumerate(train_loader):
                # ... (код кэширования данных, как был)
                if hps.if_f0 == 1:
                    (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid) = info
                else:
                    (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid) = info
                
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
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                
                if hps.if_f0 == 1:
                    cache.append((batch_idx, (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid)))
                else:
                    cache.append((batch_idx, (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid)))
            if rank == 0 and current_process_logger:
                current_process_logger.debug(f"Epoch {epoch}: Cache built with {len(cache)} batches.")
        
        if cache: # Только если кэш не пуст
             shuffle(cache) # Перемешиваем кэш каждую эпоху
        else: # Если кэш пуст (например, датасет слишком мал или пуст)
            if rank == 0 and current_process_logger:
                current_process_logger.warning(f"Epoch {epoch}: Data cache is empty. Training might not proceed if dataset is also empty.")
            data_iterator = enumerate([]) # Пустой итератор, чтобы цикл не упал

    else: # Не кэшируем
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    
    for batch_idx, batch_data in data_iterator: # batch_data это info или (idx, info) из кэша
        # Распаковка данных в зависимости от того, из кэша они или из data_loader напрямую
        if hps.if_cache_data_in_gpu and cache : # Если используем кэш и он не пуст
            original_batch_idx, info = batch_data # batch_idx здесь это индекс в перемешанном кэше
        else: # Если из DataLoader или кэш пуст (data_iterator = enumerate(train_loader))
            info = batch_data # batch_idx это индекс из enumerate(train_loader)

        if hps.if_f0 == 1:
            (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid) = info
        else:
            (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid) = info
        
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
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True) # Уже на GPU если из DataLoader или кэша

        # Calculate (код обучения как был)
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
            with autocast(enabled=False): # y_hat_mel вычисляется в float32
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax,
                )
            if hps.train.fp16_run: # Конвертируем в half если нужно
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

        if rank == 0 and current_process_logger is not None: # Логируем только в rank 0 и если логгер есть
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                # Используем длину train_loader для процента, даже если итерируемся по кэшу
                # (предполагаем, что размер кэша равен размеру эпохи)
                progress_percentage = 100.0 * (batch_idx + 1) / (len(cache) if hps.if_cache_data_in_gpu and cache else len(train_loader))

                current_process_logger.info(
                    "Train Epoch: {} [{:.0f}%] GlobalStep: {}".format(
                        epoch, progress_percentage, global_step
                    )
                )
                
                loss_mel_item = loss_mel.item() if hasattr(loss_mel, 'item') else loss_mel
                loss_kl_item = loss_kl.item() if hasattr(loss_kl, 'item') else loss_kl
                
                if loss_mel_item > 75: loss_mel_item = 75 # Ограничение для TensorBoard
                if loss_kl_item > 9: loss_kl_item = 9   # Ограничение для TensorBoard

                current_process_logger.info(f"LR: {lr}, Step: {global_step}")
                current_process_logger.info(
                    f"Losses: Disc={loss_disc.item():.3f}, Gen={loss_gen.item():.3f}, "
                    f"FM={loss_fm.item():.3f}, Mel={loss_mel_item:.3f}, KL={loss_kl_item:.3f}"
                )
                
                if writer is not None: # Убеждаемся, что writer существует (rank 0)
                    scalar_dict = {
                        "loss/g/total": loss_gen_all, "loss/d/total": loss_disc,
                        "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g,
                        "loss/g/fm": loss_fm, "loss/g/mel": loss_mel_item, "loss/g/kl": loss_kl_item
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
                        writer=writer, global_step=global_step,
                        images=image_dict, scalars=scalar_dict
                    )
        global_step += 1
    # /Run steps

    # Сохранение чекпоинтов и весов (только rank 0)
    if rank == 0 and current_process_logger is not None:
        if epoch % hps.save_every_epoch == 0:
            if hps.if_latest == 0:
                g_path = os.path.join(hps.model_dir, f"G_{global_step}.pth")
                d_path = os.path.join(hps.model_dir, f"D_{global_step}.pth")
            else: # Сохраняем как "latest" с фиксированным числом в имени
                g_path = os.path.join(hps.model_dir, "G_2333333.pth")
                d_path = os.path.join(hps.model_dir, "D_2333333.pth")
            
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_path)
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_path)

            if hps.save_every_weights == "1": # save_every_weights это строка "0" или "1"
                model_to_save = net_g.module if hasattr(net_g, "module") else net_g
                ckpt_state = model_to_save.state_dict()
                save_name = hps.name + "_e%s_s%s" % (epoch, global_step)
                
                saved_path = savee( # savee возвращает путь, куда сохранен файл
                    ckpt_state, hps.sample_rate, hps.if_f0,
                    save_name, epoch, hps.version, hps
                )
                current_process_logger.info(f"Saved small checkpoint to: {saved_path}")

        current_process_logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
        
        # Логика завершения обучения и сохранения финальной модели
        # hps.total_epoch может быть строкой из JSON, лучше конвертировать в int при чтении hps
        # или здесь: int(hps.total_epoch)
        if epoch >= int(hps.total_epoch): 
            current_process_logger.info("Target total_epoch reached. Training is done.")
            final_model_to_save = net_g.module if hasattr(net_g, "module") else net_g
            final_ckpt_state = final_model_to_save.state_dict()
            final_save_name = hps.name # Просто имя модели для финального сохранения
            
            final_saved_path = savee(
                final_ckpt_state, hps.sample_rate, hps.if_f0,
                final_save_name, epoch, hps.version, hps
            )
            current_process_logger.info(f"Saving final small checkpoint: {final_saved_path}")
            
            # Вместо os._exit, лучше просто завершить функцию,
            # чтобы DDP мог корректно завершиться через destroy_process_group.
            # Если нужно выйти из всей программы, это должно быть сделано на более высоком уровне.
            # sleep(1)
            # os._exit(2333333) # Это прервет выполнение всех процессов DDP!
                               # Удаляем это, чтобы позволить главному процессу управлять завершением.


# Эта функция будет точкой входа для DDP обучения, вызываемой из click_train_cli
def main_ddp_entry_point(hps_params_from_cli):
    # Устанавливаем переменные окружения для CUDA и определяем n_gpus
    # Это должно делаться в главном процессе, который запускает дочерние процессы DDP.
    if hasattr(hps_params_from_cli, "gpus") and hps_params_from_cli.gpus.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = hps_params_from_cli.gpus.replace("-", ",")
        n_gpus_val = len(hps_params_from_cli.gpus.split("-"))
    else: # Автоопределение GPU, если не указано
        if torch.cuda.is_available():
            n_gpus_val = torch.cuda.device_count()
            if n_gpus_val > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(n_gpus_val)))
                # Обновим hps.gpus, если он не был задан и GPU найдены
                if not hasattr(hps_params_from_cli, "gpus") or not hps_params_from_cli.gpus.strip():
                     hps_params_from_cli.gpus = "-".join(map(str, range(n_gpus_val)))
            else: # Нет CUDA GPU, но может быть MPS или XPU, или CPU
                n_gpus_val = 1 # Для DDP на CPU/MPS/XPU нужен хотя бы 1 "логический" GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = "" # Для CPU
        elif torch.backends.mps.is_available(): # Проверка MPS (Apple Silicon)
             n_gpus_val = 1
             os.environ["CUDA_VISIBLE_DEVICES"] = "" # MPS не использует CUDA_VISIBLE_DEVICES
             print("INFO: MPS backend detected. Using 1 logical GPU for DDP.")
        elif hasattr(torch, "xpu") and torch.xpu.is_available(): # Проверка XPU (Intel)
             # n_gpus_val = torch.xpu.device_count() # Уточнить API для количества XPU
             n_gpus_val = 1 # Предположим 1 для простоты, нужно уточнить
             os.environ["CUDA_VISIBLE_DEVICES"] = "" # XPU может иметь свои переменные
             print("INFO: XPU backend detected. Using 1 logical GPU for DDP (revise if multi-XPU).")
        else: # Только CPU
            n_gpus_val = 1
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("INFO: No CUDA/MPS/XPU detected. Using CPU with 1 logical GPU for DDP.")
            
    if n_gpus_val < 1:
        # Это состояние не должно достигаться с логикой выше, но на всякий случай
        print("ERROR: Number of GPUs is less than 1. Cannot start DDP training.")
        return

    # Устанавливаем MASTER_ADDR и MASTER_PORT
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # Ищем свободный порт, если utils.find_free_port() доступен, иначе фиксированный
    try:
        os.environ["MASTER_PORT"] = str(utils.find_free_port())
    except AttributeError: # Если utils.find_free_port нет
        os.environ["MASTER_PORT"] = "8665" # Запасной порт
        if n_gpus_val > 0: # Предупреждаем только если есть GPU и может быть конфликт
            print(f"WARNING: utils.find_free_port() not found. Using fixed MASTER_PORT={os.environ['MASTER_PORT']}. This might cause issues if port is busy.", file=sys.stderr)


    children = []
    # Важно: НЕ передаем логгер с FileHandler в дочерние процессы через args
    # hps_params_from_cli передается каждому процессу, он содержит всю конфигурацию
    for i in range(n_gpus_val):
        subproc = mp.Process(target=run, args=(i, n_gpus_val, hps_params_from_cli))
        children.append(subproc)
        subproc.start()

    successful_completion = True
    for i in range(n_gpus_val):
        children[i].join()
        if children[i].exitcode != 0:
            print(f"ERROR: Process {i} (rank {i}) exited with code {children[i].exitcode}", file=sys.stderr)
            successful_completion = False
    
    if successful_completion:
        print("All DDP training processes completed successfully.")
    else:
        print("One or more DDP training processes failed.")


# Этот блок if __name__ == "__main__": предназначен для прямого запуска этого скрипта,
# что обычно не происходит, когда он вызывается как часть rvc_cli.
# click_train_cli из rvc_cli/train.py должен вызывать main_ddp_entry_point.
if __name__ == "__main__":
    # Установка метода spawn должна быть одной из первых операций, если используется multiprocessing
    # и если этот скрипт может быть запущен напрямую.
    try:
        # force=True если нужно переопределить, если уже был установлен другой метод
        mp.set_start_method("spawn", force=True) 
    except RuntimeError as e:
        # Если set_start_method уже был вызван, это может вызвать ошибку.
        # Проверяем, не является ли ошибка тем, что метод уже установлен.
        if "context has already been set" not in str(e).lower():
            print(f"Error setting multiprocessing start method: {e}", file=sys.stderr)
            sys.exit(1)
        # else: # Метод уже установлен, можно продолжить (но это может быть не "spawn")
        #     print(f"Info: Multiprocessing start method already set to '{mp.get_start_method()}'.", file=sys.stderr)


    # Для прямого запуска этого файла, нужно получить hps.
    # Обычно это делается через utils.get_hparams(), который парсит sys.argv
    # или через явную загрузку конфига.
    print("INFO: Running DDP training script directly (__main__).")
    try:
        hps_main = utils.get_hparams() # Предполагается, что get_hparams() работает с sys.argv
    except Exception as e:
        print(f"CRITICAL: Failed to get hparams when running script directly: {e}", file=sys.stderr)
        print("Please ensure config.json exists and CLI arguments for get_hparams are compatible if running directly.", file=sys.stderr)
        sys.exit(1)

    # Вызываем основную точку входа DDP
    main_ddp_entry_point(hps_main)
