#!/bin/bash
set -e
UBUNTU_VERSION="ubuntu2204"
ARCH="x86_64"
echo "Шаг 1: Обновление списка пакетов и установка необходимых утилит..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends gnupg2 software-properties-common wget build-essential linux-headers-generic-hwe-22.04
echo "Шаг 2: Добавление репозитория NVIDIA CUDA..."
KEYRING_PKG="cuda-keyring_1.1-1_all.deb" # Это имя может измениться, проверьте на сайте NVIDIA
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/${ARCH}/${KEYRING_PKG}"
echo "Загрузка ключа репозитория из ${KEYRING_URL}..."
wget -q ${KEYRING_URL}
if [ ! -f ${KEYRING_PKG} ]; then
    echo "Ошибка: Не удалось загрузить пакет ключа ${KEYRING_PKG}. Проверьте URL или настройки сети."
    exit 1
fi
echo "Установка пакета ключа..."
sudo dpkg -i ${KEYRING_PKG}
echo "Очистка загруженного файла ключа..."
rm -f ${KEYRING_PKG}
echo "Шаг 3: Обновление списка пакетов после добавления репозитория CUDA..."
sudo apt-get update
echo "Шаг 4: Установка CUDA Toolkit 11.6..."
sudo apt-get install -y cuda-11-6
echo "Шаг 5: Настройка переменных окружения (системная)..."
CUDA_PROFILE_SCRIPT="/etc/profile.d/cuda-11-6.sh"
echo "Создание скрипта настройки окружения в ${CUDA_PROFILE_SCRIPT}..."
echo 'export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}' | sudo tee ${CUDA_PROFILE_SCRIPT} > /dev/null
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' | sudo tee -a ${CUDA_PROFILE_SCRIPT} > /dev/null
sudo chmod +x ${CUDA_PROFILE_SCRIPT}
echo "Установка CUDA Toolkit 11.6 завершена."
echo ""
echo "ВАЖНО:"
echo "1. Переменные окружения PATH и LD_LIBRARY_PATH были установлены глобально в ${CUDA_PROFILE_SCRIPT}."
echo "   Чтобы они применились в ТЕКУЩЕЙ сессии терминала, выполните:"
echo "   source ${CUDA_PROFILE_SCRIPT}"
echo "   Или просто откройте новый терминал."
echo "2. Настоятельно рекомендуется ПЕРЕЗАГРУЗИТЬ систему для полной инициализации драйвера NVIDIA."
echo "   Выполните: sudo reboot"
echo ""
echo "После перезагрузки проверьте установку командами:"
echo "   nvcc --version"
echo "   nvidia-smi"
exit 0
