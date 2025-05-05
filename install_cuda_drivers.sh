#!/bin/bash
set -e

# Funzione di log
log_info() {
    echo "[INFO] $1"
}

# Funzione di errore
log_error() {
    echo "[ERROR] $1"
    exit 1
}

log_info "Aggiornamento sistema"
sudo apt update && sudo apt upgrade -y

log_info "Installazione driver NVIDIA"
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

log_info "Reboot necessario dopo l'installazione dei driver"
read -p "Premi INVIO per continuare con l'installazione di CUDA dopo il riavvio..."

# Configurazione repository CUDA per Ubuntu 24.04 (CUDA 12.8)
log_info "Configurazione repository CUDA per Ubuntu 24.04 (CUDA 12.8)"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
sudo apt update

log_info "Installazione CUDA Toolkit 12.8"
sudo apt install -y cuda-toolkit-12-8

# Configura PATH e LD_LIBRARY_PATH per CUDA 12.8
log_info "Configurazione ambiente per CUDA 12.8"

# Forza solo CUDA 12.8 nel PATH e LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Ricarica le configurazioni di bash
source ~/.bashrc

log_info "Installazione PyTorch compatibile con CUDA 12.8"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

log_info "Verifica CUDA"
nvcc --version

# 1. Crea ambiente virtuale
sudo apt install python3.12-venv
python3 -m venv .venv

# 2. Attiva l'ambiente
source .venv/bin/activate

# 3. Aggiorna pip e installa PyTorch con supporto CUDA
pip install --upgrade pip

log_info "Verifica PyTorch con GPU"
python -c "import torch; print(torch.cuda.is_available())"

log_info "Installazione completata. Riavvia il sistema per applicare le modifiche."
