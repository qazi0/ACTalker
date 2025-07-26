#!/usr/bin/env python
"""
ACTalker Environment Testing Script
Verifies all dependencies and basic functionality are working properly
"""

import sys
import traceback

def test_basic_imports():
    """Test basic Python package imports"""
    print("🔍 Testing basic Python packages...")
    
    try:
        import torch
        import torchvision
        import torchaudio
        print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        
        import diffusers
        print(f"✓ Diffusers {diffusers.__version__}")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import librosa
        print(f"✓ Librosa {librosa.__version__}")
        
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        
        import omegaconf
        print(f"✓ OmegaConf {omegaconf.__version__}")
        
        import einops
        print(f"✓ Einops {einops.__version__}")
        
        import av
        print(f"✓ PyAV {av.__version__}")
        
        return True
    except Exception as e:
        print(f"✗ Basic package import failed: {e}")
        return False

def test_actalker_modules():
    """Test ACTalker project modules"""
    print("\n🔍 Testing ACTalker project modules...")
    
    try:
        from src.utils.util import seed_everything
        print("✓ src.utils.util")
        
        # Test seed setting function
        seed_everything(42)
        print("✓ seed_everything function")
        
        from src.models.audio_adapter.audio_proj import AudioProjModel
        print("✓ AudioProjModel")
        
        from src.models.audio_adapter.pose_guider import PoseGuider  
        print("✓ PoseGuider")
        
        from omegaconf import OmegaConf
        config = OmegaConf.load('config/train.yaml')
        print(f"✓ Configuration file loaded successfully (Experiment name: {config.get('exp_name', 'N/A')})")
        
        return True
    except Exception as e:
        print(f"✗ ACTalker module import failed: {e}")
        traceback.print_exc()
        return False

def test_diffusers_components():
    """Test Diffusers components"""
    print("\n🔍 Testing Diffusers core components...")
    
    try:
        from diffusers import AutoencoderKLTemporalDecoder
        print("✓ AutoencoderKLTemporalDecoder")
        
        from diffusers.schedulers import EulerDiscreteScheduler
        print("✓ EulerDiscreteScheduler")
        
        from transformers import CLIPVisionModelWithProjection, WhisperModel
        print("✓ CLIP Vision Model")
        print("✓ Whisper Model")
        
        return True
    except Exception as e:
        print(f"✗ Diffusers component import failed: {e}")
        return False

def test_audio_video_processing():
    """Test audio and video processing capabilities"""
    print("\n🔍 Testing audio/video processing components...")
    
    try:
        import cv2
        import librosa
        import moviepy
        import imageio
        import av
        
        # Create simple test data
        import numpy as np
        
        # Test image processing
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print("✓ Image data creation")
        
        # Test audio processing (create virtual audio)
        sr = 16000
        duration = 1.0
        test_audio = np.random.randn(int(sr * duration))
        print("✓ Audio data creation")
        
        return True
    except Exception as e:
        print(f"✗ Audio/video processing test failed: {e}")
        return False



def test_pretrained_models():
    """Test if pretrained models are downloaded"""
    print("\n🔍 Testing pretrained models...")
    
    try:
        import os
        from pathlib import Path
        
        # Check Stable Video Diffusion model
        svd_path = "pretrained_models/stable-video-diffusion-img2vid-xt-1-1"
        if os.path.exists(svd_path):
            # Check key files
            key_files = ["model_index.json", "unet/config.json", "vae/config.json"]
            missing_files = []
            
            for file in key_files:
                if not os.path.exists(os.path.join(svd_path, file)):
                    missing_files.append(file)
            
            if not missing_files:
                print("✓ Stable Video Diffusion model downloaded and complete")
                
                # Show model size
                size_output = os.popen(f'du -sh {svd_path}').read().strip()
                print(f"✓ Model size: {size_output.split()[0]}")
                return True
            else:
                print(f"⚠️ Stable Video Diffusion model incomplete, missing files: {missing_files}")
                print("   Please re-download the model")
                return False
        else:
            print("❌ Stable Video Diffusion model not found")
            print("   Please run: python download_models.py")
            print("   Or refer to README.md for model download instructions")
            return False
            
    except Exception as e:
        print(f"✗ Pretrained model check failed: {e}")
        return False

def test_pytorch_functionality():
    """Test PyTorch basic functionality (optimized to avoid GPU OOM)"""
    print("\n🔍 Testing PyTorch functionality...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Test basic tensor operations (using smaller tensors)
        x = torch.randn(1, 3, 64, 64)  # Reduced size to avoid memory issues
        print(f"✓ Tensor creation: {x.shape}")
        
        # Test GPU support (but don't allocate large memory on GPU)
        if torch.cuda.is_available():
            print(f"✓ GPU support: CUDA {torch.version.cuda}")
            print(f"✓ GPU device count: {torch.cuda.device_count()}")
        else:
            print("ℹ GPU unavailable, using CPU")
        
        # Test simple neural network (on CPU)
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),  # Reduced channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 5)  # Reduced output dimensions
        )
        
        with torch.no_grad():
            output = model(x)
        print(f"✓ Neural network inference: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 ACTalker environment testing started...\n")
    
    tests = [
        ("Basic Package Import", test_basic_imports),
        ("ACTalker Modules", test_actalker_modules), 
        ("Diffusers Components", test_diffusers_components),
        ("Audio/Video Processing", test_audio_video_processing),
        ("PyTorch Functionality", test_pytorch_functionality),
        ("Pretrained Models", test_pretrained_models),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} test exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! ACTalker environment setup successful!")
        print("\n📝 Next Steps:")
        print("1. Download pretrained model weights (waiting for official release)")
        print("2. Prepare test data (reference images, audio files)")
        print("3. Run inference testing")
        return True
    else:
        print("❌ Some tests failed, please check environment configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 