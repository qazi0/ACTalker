#!/bin/bash

# ACTalker Automated Installation Script
# Using tested version combination: CUDA 11.8 + PyTorch 2.0.1 + mamba-ssm 1.2.0.post1

set -e  # Exit immediately on error

echo "🚀 Starting ACTalker environment installation..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found, please install Anaconda or Miniconda first"
    exit 1
fi

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

# Install PyTorch (CUDA 11.8 compatible version)
echo "🔥 Installing PyTorch 2.0.1 (CUDA 11.8)..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install compatible numpy version
echo "📊 Installing compatible numpy version..."
pip install numpy==1.24.3

# Install diffusers and related dependencies
echo "🎨 Installing diffusers and related dependencies..."
pip install diffusers==0.29.2
pip install transformers==4.40.2
pip install accelerate==0.20.3
pip install huggingface-hub==0.34.1

# Install mamba-ssm
echo "🐍 Installing mamba-ssm..."
pip install mamba-ssm==1.2.0.post1

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

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