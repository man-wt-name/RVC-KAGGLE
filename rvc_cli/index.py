# --- rvc_cli/index.py ---
import os
import logging
import traceback
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
import shutil
import gc # Добавлен gc

logger = logging.getLogger(__name__)
now_dir = os.getcwd()
# Переменные, которые раньше были глобальными в run_rvc.py, должны быть переданы или получены из config
# index_root = "logs" # Пример
# outside_index_root = "." # Пример

def train_index_cli(exp_dir_name, version, config):
    """CLI version of train_index."""
    index_root = "logs" # Определяем здесь или получаем из config
    outside_index_root = "." # Определяем здесь или получаем из config

    exp_dir = os.path.join(index_root, exp_dir_name)
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

    if big_npy.dtype != np.float32:
        logger.warning(f"Feature dtype is {big_npy.dtype}, converting to float32.")
        big_npy = big_npy.astype(np.float32)

    if big_npy.shape[0] > 2e5:
        logger.info(f"Dataset size ({big_npy.shape[0]}) > 200k, applying MiniBatchKMeans...")
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=10000, verbose=True, batch_size=256 * config.n_cpu,
                compute_labels=False, init="random", n_init='auto'
            )
            big_npy = kmeans.fit(big_npy).cluster_centers_
            if big_npy.dtype != np.float32:
                big_npy = big_npy.astype(np.float32)
            logger.info(f"KMeans finished, reduced to {big_npy.shape[0]} centers.")
        except Exception as e:
            logger.error(f"KMeans failed: {e}\n{traceback.format_exc()}")
            logger.warning("Continuing index training with original full features.")
        finally:
             if 'kmeans' in locals(): del kmeans # Освобождаем память

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
    index_ivf = None
    try:
        index = faiss.index_factory(index_dim, f"IVF{n_ivf},Flat")
        logger.info("Training FAISS index structure...")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = max(1, int(np.power(n_ivf, 0.3)))
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

        # Create link/copy in outside_index_root
        if outside_index_root and os.path.exists(outside_index_root):
            link_func = shutil.copyfile
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
        return False
    finally:
        if index: del index
        if index_ivf: del index_ivf
        del npys, big_npy, big_npy_idx # Освобождаем память
        gc.collect()
    return True