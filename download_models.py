#!/usr/bin/env python
"""
ACTalker Pretrained Model Downloader
Downloads Stable Video Diffusion and other required models
"""

import os
import sys
from pathlib import Path

def print_banner():
    print("🚀 ACTalker Pretrained Model Downloader")
    print("=" * 50)

def check_hf_login():
    """Check Hugging Face login status"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Logged in to Hugging Face: {user_info['name']}")
        return True
    except Exception as e:
        print("❌ Not logged in to Hugging Face or authentication failed")
        return False

def download_stable_video_diffusion():
    """Download Stable Video Diffusion model"""
    from huggingface_hub import snapshot_download
    
    model_name = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
    local_path = "pretrained_models/stable-video-diffusion-img2vid-xt-1-1"
    
    print(f"\n📥 Starting download of {model_name}...")
    print("🔗 Model URL: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1")
    
    try:
        if not check_hf_login():
            print("\n🔑 Please login to Hugging Face first:")
            print("1. Run: huggingface-cli login")
            print("2. Enter your Hugging Face token")
            print("3. Re-run this script")
            return False
            
        # Check if license agreement is accepted
        print("⚠️  Note: This model requires license agreement")
        print("   Please visit: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1")
        print("   Click 'Agree and access repository' to accept the license")
        
        response = input("\nHave you accepted the license agreement and ready to download? (y/N): ")
        if response.lower() != 'y':
            print("❌ Please accept the license agreement first and try again")
            return False
        
        # Create directory
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download model
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            resume_download=True
        )
        
        print(f"✅ Model download completed!")
        print(f"📁 Model path: {model_path}")
        
        # Show model size
        os.system(f'du -sh {local_path}')
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def setup_model_directories():
    """Setup model directory structure"""
    directories = [
        "pretrained_models",
        "pretrained_models/stable-video-diffusion-img2vid-xt-1-1", 
        "pretrained_models/checkpoints",
        "outputs"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Creating directory: {dir_path}")

def main():
    print_banner()
    
    # Check required packages
    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        print("❌ Missing huggingface_hub package")
        print("Please run: pip install huggingface_hub")
        sys.exit(1)
    
    # Setup directories
    setup_model_directories()
    
    # Download guide
    print("\n📋 Model Download Guide:")
    print("1. Stable Video Diffusion (Required)")
    print("   - Base model for image-to-video generation")
    print("   - Requires agreement to Stability AI license")
    print("   - Model size: ~9.5GB")
    
    print("\n🔑 Authentication Requirements:")
    print("1. Hugging Face account needed")
    print("2. Create and configure Access Token")
    print("3. Accept model license agreement")
    
    # Check login status
    if not check_hf_login():
        print("\n🛠️  Setup Steps:")
        print("1. Visit: https://huggingface.co/settings/tokens")
        print("2. Create new Access Token (Read permission)")
        print("3. Run: huggingface-cli login")
        print("4. Enter your token")
        print("5. Re-run this script")
        sys.exit(1)
    
    # Download models
    print("\n🚀 Starting model download...")
    success = download_stable_video_diffusion()
    
    if success:
        print("\n🎉 All models downloaded successfully!")
        print("\n📝 Next Steps:")
        print("1. Check configuration files: config/train.yaml and config/inference.yaml")
        print("2. Update model paths to point to downloaded models")
        print("3. Run environment test: python test_environment.py")
        print("4. Start inference testing")
    else:
        print("\n❌ Model download failed, please check network connection and authentication status")
        sys.exit(1)

if __name__ == "__main__":
    main() 