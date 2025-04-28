#!/bin/bash

echo "Останавливаем службы, связанные с NVIDIA (если запущены)..."
sudo systemctl stop nvidia-persistenced || true # Игнорируем ошибку, если служба не найдена

echo "Удаляем пакеты CUDA и NVIDIA с помощью apt..."
# Сначала удаляем основные пакеты CUDA и драйверы
# Флаг -y автоматически отвечает 'yes' на все запросы apt
sudo apt-get purge -y --autoremove 'cuda*' '*cublas*' '*cufft*' '*curand*' \
 '*cusolver*' '*cusparse*' '*npp*' '*nvjpeg*' 'nvidia-*' \
 'libnvidia-*' 'libcuda*' 'libcudnn*'

echo "Очищаем кэш apt..."
sudo apt-get clean

echo "Удаляем остаточные конфигурационные файлы и директории..."
# Будьте осторожны с rm -rf!
sudo rm -rf /usr/local/cuda*
sudo rm -f /etc/apt/sources.list.d/cuda*
sudo rm -f /etc/apt/sources.list.d/nvidia*
sudo rm -f /etc/profile.d/cuda.sh # Пример файла окружения

echo "Обновляем список пакетов..."
sudo apt-get update

echo "Проверяем зависимости, которые больше не нужны..."
# Еще раз запускаем autoremove на всякий случай
sudo apt-get autoremove -y

echo "Процесс удаления завершен."
echo "Настоятельно рекомендуется перезагрузить систему:"
echo "sudo reboot"