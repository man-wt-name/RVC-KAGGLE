# Файл: infer/lib/train/utils.py
import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import shutil # Добавлен импорт shutil
import numpy as np
import torch
from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

# Настраиваем корневой логгер, чтобы видеть все сообщения по умолчанию
logging.basicConfig(stream=sys.stdout, level=logging.INFO, # Уровень INFO для консоли по умолчанию
                    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
# logger = logging.getLogger(__name__) # Логгер этого модуля - убрано, так как get_logger создает свой

# --- ИСПРАВЛЕННАЯ get_logger ---
def get_logger(model_dir, filename="train.log"):
    # Используем имя директории модели как имя логгера для уникальности
    logger_name = os.path.basename(model_dir)
    # Получаем логгер с этим именем. Если он уже существует, возвращаем его.
    logger = logging.getLogger(logger_name)

    # Устанавливаем уровень логгера (например, DEBUG, чтобы захватывать все сообщения)
    logger.setLevel(logging.DEBUG)

    # Предотвращаем добавление обработчиков, если они уже есть
    if not logger.handlers:
        # Форматтер для логов
        formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

        # Обработчик для записи в файл
        if not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir)
            except OSError as e:
                print(f"Error creating directory {model_dir}: {e}. Logs will not be saved to file.")

        try:
            fh = logging.FileHandler(os.path.join(model_dir, filename), encoding='utf-8') # Указываем кодировку
            fh.setLevel(logging.DEBUG) # Уровень для файла может быть DEBUG
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Error creating file handler for {os.path.join(model_dir, filename)}: {e}. Logs will not be saved to file.")

        # Обработчик для вывода в консоль (stdout)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO) # Уровень для консоли можно оставить INFO
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # Предотвращаем передачу сообщений корневому логгеру, если он уже настроен
        logger.propagate = False

    return logger
# --- КОНЕЦ ИСПРАВЛЕННОЙ get_logger ---

def load_checkpoint_d(checkpoint_path, combd, sbd, optimizer=None, load_opt=1):
    # Получаем логгер для текущего контекста (если utils.py используется в разных местах)
    current_logger = logging.getLogger(__name__)
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    ##################
    def go(model, bkey):
        saved_state_dict = checkpoint_dict[bkey]
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():  # 模型需要的shape
            try:
                # Проверка формы перед присваиванием
                if k in saved_state_dict and saved_state_dict[k].shape == state_dict[k].shape:
                    new_state_dict[k] = saved_state_dict[k]
                elif k in saved_state_dict:
                     current_logger.warning(
                        "Shape mismatch for key %s. Model needs %s, checkpoint has %s. Using initial value.",
                        k, state_dict[k].shape, saved_state_dict[k].shape
                     )
                     new_state_dict[k] = v # Используем исходное значение модели
                else:
                    current_logger.info("%s not found in checkpoint, using initial value.", k) # pretrain缺失的
                    new_state_dict[k] = v  # 模型自带的随机值
            except Exception as e:
                current_logger.error(f"Error loading key {k}: {e}")
                new_state_dict[k] = v # Используем исходное значение в случае ошибки
        if hasattr(model, "module"):
            model.module.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=False)
        return model

    go(combd, "combd")
    model = go(sbd, "sbd")
    #############
    current_logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
        and "optimizer" in checkpoint_dict # Проверяем наличие ключа
    ):
        try:
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
        except Exception as e:
            current_logger.warning(f"Failed to load optimizer state: {e}. Initializing optimizer from scratch.")
    elif optimizer is not None and load_opt == 1:
         current_logger.warning("Optimizer state not found in checkpoint. Initializing optimizer from scratch.")


    current_logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    current_logger = logging.getLogger(__name__)
    assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"
    try:
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
         current_logger.error(f"Failed to load checkpoint file {checkpoint_path}: {e}")
         raise

    saved_state_dict = checkpoint_dict.get("model") # Используем .get для безопасного доступа
    if saved_state_dict is None:
         # Возможно, это старый формат чекпоинта без ключа 'model'
         saved_state_dict = checkpoint_dict
         current_logger.warning("Checkpoint does not contain a 'model' key. Assuming the entire dict is the state_dict.")
         if not isinstance(saved_state_dict, dict):
              raise ValueError(f"Checkpoint format unknown or corrupt: {checkpoint_path}")

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    model_keys = set(state_dict.keys())
    ckpt_keys = set(saved_state_dict.keys())

    # Ключи, присутствующие и в модели, и в чекпоинте
    intersect_keys = model_keys.intersection(ckpt_keys)
    # Ключи, отсутствующие в чекпоинте, но присутствующие в модели
    missing_keys = model_keys - ckpt_keys
    # Ключи, присутствующие в чекпоинте, но отсутствующие в модели
    unexpected_keys = ckpt_keys - model_keys

    # Загружаем совпадающие ключи, проверяя форму
    for k in intersect_keys:
        if saved_state_dict[k].shape == state_dict[k].shape:
            new_state_dict[k] = saved_state_dict[k]
        else:
            current_logger.warning(
                f"Shape mismatch for key {k}. Model needs {state_dict[k].shape}, checkpoint has {saved_state_dict[k].shape}. Using initial value."
            )
            new_state_dict[k] = state_dict[k] # Используем исходное значение модели

    # Добавляем ключи, которых не было в чекпоинте (используем инициализацию модели)
    for k in missing_keys:
        new_state_dict[k] = state_dict[k]
        current_logger.info(f"Key '{k}' not found in checkpoint, using initial value.")

    # Сообщаем о ключах, которые были в чекпоинте, но не используются моделью
    if unexpected_keys:
        current_logger.warning(f"Unexpected keys found in checkpoint and ignored: {list(unexpected_keys)}")

    # Загружаем state_dict
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    current_logger.info("Loaded model weights from checkpoint.")

    iteration = checkpoint_dict.get("iteration", 0) # Безопасное получение итерации
    learning_rate = checkpoint_dict.get("learning_rate", None) # Безопасное получение LR

    if optimizer is not None and load_opt == 1 and "optimizer" in checkpoint_dict:
        try:
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
            current_logger.info("Loaded optimizer state from checkpoint.")
        except Exception as e:
            current_logger.warning(f"Failed to load optimizer state: {e}. Initializing optimizer from scratch.")
    elif optimizer is not None and load_opt == 1:
        current_logger.warning("Optimizer state not found in checkpoint. Initializing optimizer from scratch.")

    current_logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    current_logger = logging.getLogger(__name__)
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    current_logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    try:
        torch.save(
            {
                "model": state_dict,
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "learning_rate": learning_rate,
            },
            checkpoint_path,
        )
        current_logger.info(f"Successfully saved checkpoint: {checkpoint_path}")
    except Exception as e:
        current_logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")

def save_checkpoint_d(combd, sbd, optimizer, learning_rate, iteration, checkpoint_path):
    current_logger = logging.getLogger(__name__)
     # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    current_logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(combd, "module"):
        state_dict_combd = combd.module.state_dict()
    else:
        state_dict_combd = combd.state_dict()
    if hasattr(sbd, "module"):
        state_dict_sbd = sbd.module.state_dict()
    else:
        state_dict_sbd = sbd.state_dict()
    try:
        torch.save(
            {
                "combd": state_dict_combd,
                "sbd": state_dict_sbd,
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "learning_rate": learning_rate,
            },
            checkpoint_path,
        )
        current_logger.info(f"Successfully saved D checkpoint: {checkpoint_path}")
    except Exception as e:
         current_logger.error(f"Failed to save D checkpoint {checkpoint_path}: {e}")

def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    current_logger = logging.getLogger(__name__)
    try:
        for k, v in scalars.items():
            writer.add_scalar(k, v, global_step)
        for k, v in histograms.items():
            writer.add_histogram(k, v, global_step)
        for k, v in images.items():
            writer.add_image(k, v, global_step, dataformats="HWC")
        for k, v in audios.items():
            writer.add_audio(k, v, global_step, audio_sampling_rate)
    except Exception as e:
        current_logger.error(f"Error summarizing data for TensorBoard: {e}")

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    current_logger = logging.getLogger(__name__)
    try:
        f_list = glob.glob(os.path.join(dir_path, regex))
        if not f_list:
            # Если ищем G_*.pth и его нет, попробуем G_2333333.pth
            if regex == "G_*.pth":
                 fallback_path = os.path.join(dir_path, "G_2333333.pth")
                 if os.path.exists(fallback_path):
                     current_logger.debug(f"Found latest G checkpoint: {fallback_path}")
                     return fallback_path
            # Если ищем D_*.pth и его нет, попробуем D_2333333.pth
            elif regex == "D_*.pth":
                 fallback_path = os.path.join(dir_path, "D_2333333.pth")
                 if os.path.exists(fallback_path):
                      current_logger.debug(f"Found latest D checkpoint: {fallback_path}")
                      return fallback_path
            # Иначе возвращаем пустую строку или возбуждаем исключение
            current_logger.warning(f"No checkpoints found matching {regex} in {dir_path}")
            return "" # Или можно: raise FileNotFoundError(f"No checkpoints found...")

        # Исправлено извлечение номера эпохи для правильной сортировки
        f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f.split("_")[-1].split(".")[0])) or 0))
        x = f_list[-1]
        current_logger.debug(f"Found latest checkpoint: {x}")
        return x
    except Exception as e:
        current_logger.error(f"Error finding latest checkpoint in {dir_path} with regex {regex}: {e}")
        return "" # Возвращаем пустую строку в случае ошибки

def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    current_logger = logging.getLogger(__name__)
    try:
        if not MATPLOTLIB_FLAG:
            import matplotlib
            matplotlib.use("Agg")
            MATPLOTLIB_FLAG = True
            mpl_logger = logging.getLogger("matplotlib")
            mpl_logger.setLevel(logging.WARNING)
        import matplotlib.pylab as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 2))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig) # Закрываем фигуру явно
        return data
    except Exception as e:
        current_logger.error(f"Error plotting spectrogram: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8) # Возвращаем пустой массив

def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    current_logger = logging.getLogger(__name__)
    try:
        if not MATPLOTLIB_FLAG:
            import matplotlib
            matplotlib.use("Agg")
            MATPLOTLIB_FLAG = True
            mpl_logger = logging.getLogger("matplotlib")
            mpl_logger.setLevel(logging.WARNING)
        import matplotlib.pylab as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
        )
        fig.colorbar(im, ax=ax)
        xlabel = "Decoder timestep"
        if info is not None:
            xlabel += "\n\n" + info
        plt.xlabel(xlabel)
        plt.ylabel("Encoder timestep")
        plt.tight_layout()

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig) # Закрываем фигуру явно
        return data
    except Exception as e:
        current_logger.error(f"Error plotting alignment: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8) # Возвращаем пустой массив

def load_wav_to_torch(full_path):
    current_logger = logging.getLogger(__name__)
    try:
        sampling_rate, data = read(full_path)
        # Проверка на случай, если wavfile.read возвращает неверный тип
        if not isinstance(data, np.ndarray):
            raise TypeError(f"wavfile.read returned non-numpy array for {full_path}")
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate
    except Exception as e:
        current_logger.error(f"Error loading wav file {full_path}: {e}")
        raise

def load_filepaths_and_text(filename, split="|"):
    current_logger = logging.getLogger(__name__)
    try:
        with open(filename, encoding="utf-8") as f:
            filepaths_and_text = [line.strip().split(split) for line in f if line.strip()] # Добавлена проверка на пустые строки
    except UnicodeDecodeError:
        current_logger.warning(f"File {filename} is not UTF-8 encoded, trying default encoding.")
        try:
            with open(filename) as f:
                filepaths_and_text = [line.strip().split(split) for line in f if line.strip()]
        except Exception as e:
             current_logger.error(f"Failed to read filelist {filename}: {e}")
             return [] # Возвращаем пустой список в случае ошибки
    except FileNotFoundError:
        current_logger.error(f"Filelist {filename} not found.")
        return []
    except Exception as e:
        current_logger.error(f"Failed to process filelist {filename}: {e}")
        return []

    # Проверка корректности формата строк
    valid_lines = []
    expected_len = 0
    if filepaths_and_text:
         first_line_len = len(filepaths_and_text[0])
         if first_line_len == 5: expected_len = 5 # F0 model
         elif first_line_len == 3: expected_len = 3 # non-F0 model
         else:
              current_logger.warning(f"Unexpected number of columns in first line of {filename}: {first_line_len}. Cannot determine expected format.")
              expected_len = first_line_len # Попробуем использовать первую строку как шаблон

    for i, line in enumerate(filepaths_and_text):
        if len(line) == expected_len:
            valid_lines.append(line)
        else:
            # Не выводим содержимое строки в лог по умолчанию для конфиденциальности
            current_logger.warning(f"Skipping line {i+1} in {filename}: expected {expected_len} columns, got {len(line)}.")

    return valid_lines


def get_hparams(init=True):
    current_logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-se", "--save_every_epoch", type=int, required=True, help="checkpoint save frequency (epoch)")
    parser.add_argument("-te", "--total_epoch", type=int, required=True, help="total_epoch")
    parser.add_argument("-pg", "--pretrainG", type=str, default="", help="Pretrained Generator path")
    parser.add_argument("-pd", "--pretrainD", type=str, default="", help="Pretrained Discriminator path")
    parser.add_argument("-g", "--gpus", type=str, default="0", help="split by -")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("-e", "--experiment_dir", type=str, required=True, help="experiment dir") # -m
    parser.add_argument("-sr", "--sample_rate", type=str, required=True, help="sample rate, 32k/40k/48k")
    parser.add_argument("-sw", "--save_every_weights", type=str, default="0", help="save the extracted model in weights directory when saving checkpoints")
    parser.add_argument("-v", "--version", type=str, required=True, help="model version")
    parser.add_argument("-f0", "--if_f0", type=int, required=True, help="use f0 as one of the inputs of the model, 1 or 0")
    parser.add_argument("-l", "--if_latest", type=int, required=True, help="if only save the latest G/D pth file, 1 or 0")
    parser.add_argument("-c", "--if_cache_data_in_gpu", type=int, required=True, help="if caching the dataset in GPU memory, 1 or 0")
    args = parser.parse_args()

    exp_dir_normalized = os.path.normpath(args.experiment_dir)
    name = os.path.basename(exp_dir_normalized)

    # Указываем путь к logs относительно текущей директории
    logs_dir = os.path.join(os.getcwd(), "logs")
    experiment_dir_path = os.path.join(logs_dir, name)

    config_save_path = os.path.join(experiment_dir_path, "config.json")

    if not os.path.exists(config_save_path):
        current_logger.error(f"Config file not found at {config_save_path}. Make sure the experiment directory exists and preprocessing was run.")
        sys.exit(1)

    try:
        with open(config_save_path, "r", encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        current_logger.error(f"Failed to load config file {config_save_path}: {e}")
        sys.exit(1)

    hparams = HParams(**config)
    hparams.model_dir = experiment_dir_path
    hparams.experiment_dir = experiment_dir_path
    hparams.save_every_epoch = args.save_every_epoch
    hparams.name = name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.version = args.version
    hparams.gpus = args.gpus
    hparams.train.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    hparams.data.training_files = os.path.join(experiment_dir_path, "filelist.txt").replace("\\", "/")

    return hparams


def get_hparams_from_dir(model_dir):
    current_logger = logging.getLogger(__name__)
    config_save_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_save_path):
         current_logger.error(f"Config file not found at {config_save_path}")
         return HParams() # Возвращаем пустой HParams
    try:
        with open(config_save_path, "r", encoding='utf-8') as f:
            data = f.read()
        config = json.loads(data)
        hparams = HParams(**config)
        hparams.model_dir = model_dir
        return hparams
    except Exception as e:
        current_logger.error(f"Failed to load config from {config_save_path}: {e}")
        return HParams()

def get_hparams_from_file(config_path):
    current_logger = logging.getLogger(__name__)
    if not os.path.exists(config_path):
         current_logger.error(f"Config file not found at {config_path}")
         return HParams()
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            data = f.read()
        config = json.loads(data)
        hparams = HParams(**config)
        return hparams
    except Exception as e:
        current_logger.error(f"Failed to load config from {config_path}: {e}")
        return HParams()

def check_git_hash(model_dir):
    current_logger = logging.getLogger(__name__)
    source_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # Путь к корню проекта RVC
    git_dir = os.path.join(source_dir, ".git")
    if not os.path.exists(git_dir):
        current_logger.warning(
            "'{}/.git' directory not found. Cannot check git hash.".format(source_dir)
        )
        return

    try:
        # Используем check_output для получения вывода и обработки ошибок
        cur_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_dir, stderr=subprocess.STDOUT).decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        current_logger.warning(f"Failed to get current git hash: {e.output.decode('utf-8', errors='ignore')}")
        return
    except FileNotFoundError:
        current_logger.warning("Git command not found. Cannot check git hash.")
        return
    except Exception as e:
         current_logger.warning(f"An unexpected error occurred while checking git hash: {e}")
         return


    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f: # Добавляем encoding
                saved_hash = f.read().strip()
            if saved_hash != cur_hash:
                current_logger.warning(
                    "Git hash mismatch! Saved: {}(...) vs Current: {}(...). Model might be incompatible.".format(
                        saved_hash[:8], cur_hash[:8]
                    )
                )
        except Exception as e:
             current_logger.warning(f"Failed to read saved git hash from {path}: {e}")
    else:
        try:
            os.makedirs(model_dir, exist_ok=True) # Убедимся, что директория есть
            with open(path, "w", encoding='utf-8') as f: # Добавляем encoding
                f.write(cur_hash)
        except Exception as e:
            current_logger.warning(f"Failed to write current git hash to {path}: {e}")


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        # Добавим проверку на существование атрибута для избежания ошибок
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()