#!/bin/bash

# Выходить немедленно, если команда завершается с ошибкой
set -e

# --- КОНФИГУРАЦИЯ (измените при необходимости) ---
# Определите версию Ubuntu и архитектуру (примеры)
# Для Ubuntu 22.04 (Jammy) x86_64:
UBUNTU_VERSION="ubuntu2204"
ARCH="x86_64"
# Для Ubuntu 20.04 (Focal) x86_64:
# UBUNTU_VERSION="ubuntu2004"
# ARCH="x86_64"

# --- ОСНОВНОЙ СКРИПТ ---

echo "Шаг 1: Обновление списка пакетов и установка необходимых утилит..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends gnupg2 software-properties-common wget build-essential linux-headers-generic-hwe-22.04

echo "Шаг 2: Добавление репозитория NVIDIA CUDA..."

# Установка ключа репозитория NVIDIA
# Используем временный файл для ключа
KEYRING_PKG="cuda-keyring_1.1-1_all.deb" # Это имя может измениться, проверьте на сайте NVIDIA
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/${ARCH}/${KEYRING_PKG}"

echo "Загрузка ключа репозитория из ${KEYRING_URL}..."
wget -q ${KEYRING_URL}

if [ ! -f ${KEYRING_PKG} ]; then
    echo "Ошибка: Не удалось загрузить пакет ключа ${KEYRING_PKG}. Проверьте URL или настройки сети."
    exit 1
fi

echo "Установка пакета ключа..."
# Флаг -y не нужен для dpkg, но используем DEBIAN_FRONTEND для неинтерактивности apt (если dpkg вызовет apt)
sudo dpkg -i ${KEYRING_PKG}

# Добавляем сам репозиторий (пакет ключа обычно делает это, но можно и явно)
# Если пакет ключа не добавил репозиторий, раскомментируйте следующие строки:
# REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/${ARCH}/"
# echo "Добавление CUDA репозитория: ${REPO_URL}"
# sudo add-apt-repository "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] ${REPO_URL} /"

echo "Очистка загруженного файла ключа..."
rm -f ${KEYRING_PKG}

echo "Шаг 3: Обновление списка пакетов после добавления репозитория CUDA..."
sudo apt-get update

echo "Шаг 4: Установка CUDA Toolkit 11.6..."
# Устанавливаем мета-пакет cuda-11-6, который должен подтянуть нужные компоненты (toolkit, driver)
# Флаг -y отвечает 'yes' на все запросы
sudo apt-get install -y cuda-11-6

echo "Шаг 5: Настройка переменных окружения (системная)..."
# Создаем файл для profile.d, чтобы переменные были доступны всем пользователям
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