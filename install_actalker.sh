#!/bin/bash

# ACTalker Automated Installation Script
# Using tested version combination: CUDA 11.8 + PyTorch 2.0.1 + mamba-ssm 1.2.0.post1

set -e  # Exit immediately on error

apt update && apt install zsh unzip -y


echo "🚀 Starting ACTalker environment installation..."

CONDA_DIR="$HOME/miniconda3"

# Install Miniconda if conda is not available
if ! command -v conda &>/dev/null; then
    echo "[INFO] Installing Miniconda..."
    ARCH=$(uname -m)
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    curl -sSL "$URL" -o miniconda.sh
    bash miniconda.sh -b -u -p "$CONDA_DIR"
    rm miniconda.sh
    export PATH="$CONDA_DIR/bin:$PATH"
fi

# Accept ToS for channels
echo "[INFO] Accepting Anaconda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Source Conda
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: NVIDIA driver not found, please ensure NVIDIA GPU driver is installed"
    exit 1
fi

# Check FFmpeg and libx264
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  Warning: FFmpeg not found, attempting to install..."
    # Try to install FFmpeg and x264 via conda
    conda install -c conda-forge ffmpeg x264 -y
    if ! command -v ffmpeg &> /dev/null; then
        echo "❌ Error: Failed to install FFmpeg. Please install manually:"
        echo "   Ubuntu/Debian: sudo apt install ffmpeg libx264-dev"
        echo "   CentOS/RHEL: sudo yum install ffmpeg x264-devel"
        exit 1
    fi
fi

echo "✅ Detected conda, NVIDIA driver, FFmpeg, and libx264"

# Create conda environment
echo "📦 Creating actalker conda environment..."
conda create -n actalker python=3.10 -y

# Activate environment
echo "🔧 Activating actalker environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate actalker

# Install CUDA 11.8 toolkit
echo "⚡ Installing CUDA 11.8 toolkit..."
conda install cudatoolkit=11.8 -c conda-forge -y

# Set environment variables
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install PyTorch (CUDA 11.8 compatible version)
echo "🔥 Installing PyTorch 2.0.1 (CUDA 11.8)..."
uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install compatible numpy version
echo "📊 Installing compatible numpy version..."
uv pip install numpy==1.24.3

# Install diffusers and related dependencies
echo "🎨 Installing diffusers and related dependencies..."
uv pip install diffusers==0.29.2
uv pip install transformers==4.40.2
uv pip install accelerate==0.20.3
uv pip install huggingface-hub==0.34.1

# Install mamba-ssm
echo "🐍 Installing mamba-ssm..."
pip install mamba-ssm==1.2.0.post1

# Install other dependencies
echo "📚 Installing other dependencies..."
uv pip install -r requirements.txt

python download_models.py
# curl 'https://southeastasia1-mediap.svc.ms/transform/zip?cs=fFNQTw' \
#   -X POST \
#   -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:142.0) Gecko/20100101 Firefox/142.0' \
#   -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
#   -H 'Accept-Language: en-US,en;q=0.5' \
#   -H 'Accept-Encoding: gzip, deflate, br, zstd' \
#   -H 'Content-Type: application/x-www-form-urlencoded' \
#   -H 'Origin: https://hkustconnect-my.sharepoint.com' \
#   -H 'Connection: keep-alive' \
#   -H 'Upgrade-Insecure-Requests: 1' \
#   -H 'Sec-Fetch-Dest: iframe' \
#   -H 'Sec-Fetch-Mode: navigate' \
#   -H 'Sec-Fetch-Site: cross-site' \
#   -H 'Priority: u=4' \
#   --data-raw 'zipFileName=ICCV2025_ACTalker.zip&guid=225533a1-7740-466a-a9ec-c258e38c4b6a&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%22ICCV2025_ACTalker%22%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Fhkustconnect-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21HcEGtPob0kaTxC403dA12kdZPxuH4HxKpB3pIxX3FXSRAi7DENf_TLwsSHFC4X1e%2Fitems%2F01BOPCLLFWZAMCMD5EQRBYYVYPQLLBPIGA%3Fversion%3DPublished' -o checkpoints.zip

# unzip checkpoints.zip -d pretrained_models/checkpoints/
# rm -rf checkpoints.zip
mv pretrained_models/checkpoints/ICCV*/* pretrained_models/checkpoints/
rm -rf pretrained_models/checkpoints/ICCV*

# curl 'https://hkustconnect-my.sharepoint.com/personal/fhongac_connect_ust_hk/_layouts/15/download.aspx?UniqueId=d911d1c1%2D560c%2D4153%2Dbf40%2D2bf90caff7fb' \
#   -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:142.0) Gecko/20100101 Firefox/142.0' \
#   -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
#   -H 'Accept-Language: en-US,en;q=0.5' \
#   -H 'Accept-Encoding: gzip, deflate, br, zstd' \
#   -H 'Referer: https://hkustconnect-my.sharepoint.com/personal/fhongac_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Ffhongac%5Fconnect%5Fust%5Fhk%2FDocuments%2FCheckpoint%2FICCV2025%5FACTalker&ga=1' \
#   -H 'Upgrade-Insecure-Requests: 1' \
#   -H 'Sec-Fetch-Dest: iframe' \
#   -H 'Sec-Fetch-Mode: navigate' \
#   -H 'Sec-Fetch-Site: same-origin' \
#   -H 'Connection: keep-alive' \
#   -H 'Cookie: FedAuth=' -o enhance.zip

# unzip enhance.zip
# rm -rf enhance.zip

# Verify installation
echo "🔍 Verifying installation..."
echo "Checking FFmpeg..."
ffmpeg -version | head -1

echo "Checking PyTorch..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}')"

echo "Checking Mamba SSM..."
python -c "
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print('✅ Mamba SSM imported successfully')
except ImportError as e:
    print(f'❌ Mamba SSM import failed: {e}')
    exit(1)
"

echo "Checking ACTalker components..."
python -c "
try:
    from src.models.base.mamba_layer import MAMBA_AVAILABLE
    print(f'✅ ACTalker Mamba components available: {MAMBA_AVAILABLE}')
except ImportError as e:
    print(f'❌ ACTalker component import failed: {e}')
    exit(1)
"

echo "🎉 ACTalker environment installation completed!"
echo ""
echo "📋 Installation Summary:"
echo "  - Environment name: actalker" 
echo "  - Python: 3.10"
echo "  - CUDA: 11.8"
echo "  - PyTorch: 2.0.1+cu118"
echo "  - mamba-ssm: 1.2.0.post1"
echo "  - FFmpeg: $(ffmpeg -version | head -1 | grep -o 'version [0-9.]*' | cut -d' ' -f2)"
echo ""
echo "🚀 Usage:"
echo "  conda activate actalker"
echo "  python Inference.py --config config/inference.yaml --ref assets/ref.jpg --audio assets/audio.mp3 --video assets/video.mp4 --mode 2"
echo ""
echo "⚠️  Important Notes:"
echo "  - Ensure GPU VRAM >= 24GB"
echo "  - Inference takes approximately 6-8 minutes (H100)"
echo "  - Check CUDA version compatibility if you encounter issues" 