# rvc_cli/helpers.py
import os
import sys
import logging
import traceback
import threading
from subprocess import Popen, PIPE, STDOUT
from time import sleep

logger = logging.getLogger(__name__)
now_dir = os.getcwd() # Определяем здесь или передаем

def run_process(cmd, config):
    """Helper to run subprocess and stream output."""
    logger.info(f"Executing: {cmd}")
    try:
        # Используем config.python_cmd, если команда начинается с "python" или ""
        actual_cmd = cmd
        if cmd.strip().startswith("python ") or cmd.strip().startswith('"python" '):
             actual_cmd = cmd.replace("python", config.python_cmd, 1).replace('"python"', f'"{config.python_cmd}"', 1)
             logger.debug(f"Adjusted python command: {actual_cmd}")

        p = Popen(actual_cmd, shell=True, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, universal_newlines=True, cwd=now_dir, env=os.environ)
        if p.stdout:
            for line in iter(p.stdout.readline, ''):
                print(line.strip())
        p.wait()
        if p.returncode != 0:
            logger.error(f"Command '{actual_cmd}' failed with return code {p.returncode}")
        else:
             logger.info(f"Command '{actual_cmd}' finished successfully.")
        return p.returncode
    except Exception as e:
        logger.error(f"Failed to execute command '{cmd}': {e}")
        logger.error(traceback.format_exc())
        return -1

def run_process_parallel(cmds, config):
    """Helper to run multiple subprocesses in parallel and stream output."""
    processes = []
    outputs = {}
    threads = []
    all_successful = True # Флаг успеха

    def reader_thread(p, pid, output_list):
        if p.stdout:
            for line in iter(p.stdout.readline, ''):
                output_list.append(line)
                print(f"[PID {pid}]: {line.strip()}")
        rc = p.wait()
        logger.info(f"Process PID {pid} finished with return code {rc}.")
        # Не меняем флаг all_successful здесь, проверяем в основном потоке

    logger.info("Starting parallel processes...")
    for i, cmd in enumerate(cmds):
        logger.info(f"Executing [{i+1}/{len(cmds)}]: {cmd}")
        try:
            actual_cmd = cmd
            if cmd.strip().startswith("python ") or cmd.strip().startswith('"python" '):
                actual_cmd = cmd.replace("python", config.python_cmd, 1).replace('"python"', f'"{config.python_cmd}"', 1)
                logger.debug(f"Adjusted python command: {actual_cmd}")

            p = Popen(actual_cmd, shell=True, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, universal_newlines=True, cwd=now_dir, env=os.environ)
            pid = p.pid
            outputs[pid] = []
            thread = threading.Thread(target=reader_thread, args=(p, pid, outputs[pid]))
            thread.start()
            threads.append(thread)
            processes.append(p)
        except Exception as e:
            logger.error(f"Failed to start command '{cmd}': {e}")
            all_successful = False # Отмечаем неудачу при запуске

    for thread in threads:
        thread.join()

    # Проверяем коды возврата после завершения всех потоков
    failed_cmds = []
    for i, p in enumerate(processes):
        if p.returncode != 0:
            logger.error(f"Command '{cmds[i]}' (PID {p.pid}) failed with return code {p.returncode}")
            failed_cmds.append(cmds[i])
            all_successful = False # Отмечаем неудачу

    if all_successful:
         logger.info("All parallel processes finished successfully.")
    else:
         logger.warning(f"One or more parallel processes failed: {failed_cmds}")

    return all_successful

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

# Функции if_done, if_done_multi можно оставить здесь или удалить,
# если run_process/run_process_parallel теперь блокирующие