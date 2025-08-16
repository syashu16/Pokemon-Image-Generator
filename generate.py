#!/usr/bin/env python3
"""
🎨 Pokemon LoRA Generator - Simple Image Generation Script
Author: syashu16
Description: Easy-to-use script for generating Pokemon-style images
"""

import argparse
import os
import sys
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_environment():
    """Setup environment variables for optimal performance"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['HF_HOME'] = str(project_root / 'models' / 'cache')
    os.environ['TRANSFORMERS_CACHE'] = str(project_root / 'models' / 'cache' / 'transformers')
    
    # Create directories
    (project_root / 'models' / 'cache').mkdir(parents=True, exist_ok=True)
    (project_root / 'outputs').mkdir(exist_ok=True)

def check_system():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Please upgrade Python.")
        sys.exit(1)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 3.5:
            print("⚠️ Warning: GPU has less than 4GB VRAM. Consider using --device cpu")
    else:
        print("⚠️ No GPU detected. Will use CPU (slower generation)")
    
    print("✅ System check complete!")

def download_model():
    """Download the Pokemon LoRA model if not present"""
    model_path = project_root / 'models' / 'pokemon_lora_model'
    
    if model_path.exists():
        print("✅ Model already downloaded!")
        return str(model_path)
    
    print("📥 Downloading Pokemon LoRA model...")
    print("This may take a few minutes on first run...")
    
    try:
        # In a real implementation, you'd download from Hugging Face Hub
        # For now, we'll use the local trained model
        import shutil
        
        # Check if model exists in D drive cache
        d_drive_model = Path("D:/ai_models_cache/pokemon_lora_model")
        if d_drive_model.exists():
            print("📋 Copying model from D drive cache...")
            shutil.copytree(d_drive_model, model_path)
            print("✅ Model copied successfully!")
        else:
            print("❌ Model not found. Please train the model first using app.py")
            print("💡 Run: python app.py")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)
    
    return str(model_path)

def generate_pokemon(prompt, model_path, args):
    """Generate Pokemon image using the trained LoRA model"""
    print(f"🎨 Generating: '{prompt}'")
    
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        from peft import PeftModel
        import torch
        from PIL import Image
        import numpy as np
        
        # Setup device
        device = torch.device(args.device if args.device != 'auto' else 
                            ('cuda' if torch.cuda.is_available() else 'cpu'))
        dtype = torch.float32 if device.type == 'cpu' else torch.float16
        
        print(f"🎯 Using device: {device}")
        
        # Load base pipeline
        print("📥 Loading Stable Diffusion pipeline...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=os.environ['TRANSFORMERS_CACHE']
        )
        
        # Load LoRA weights
        print("🔧 Loading Pokemon LoRA weights...")
        try:
            pipeline.load_lora_weights(model_path)
            print("✅ LoRA weights loaded!")
        except Exception as e:
            print(f"⚠️ Could not load LoRA weights: {e}")
            print("Using base Stable Diffusion model...")
        
        # Move to device and optimize
        pipeline = pipeline.to(device)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        # Enable memory optimizations
        pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
        
        # Generate image
        print("🎨 Generating image...")
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
        with torch.no_grad():
            result = pipeline(
                prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
                height=args.size,
                width=args.size,
                negative_prompt="blurry, bad quality, distorted, deformed, ugly"
            )
            image = result.images[0]
        
        # Check image quality
        img_array = np.array(image)
        brightness = np.mean(img_array)
        
        if brightness < 20:
            print("⚠️ Generated image appears dark. Try different prompt or seed.")
        else:
            print("✅ Image generated successfully!")
        
        return image
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="🎨 Pokemon LoRA Generator - Create Pokemon-style images with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --prompt "a cute fire pokemon with orange fur"
  python generate.py --prompt "a water pokemon swimming" --steps 30 --seed 42
  python generate.py --prompt "electric pokemon" --device cpu --size 256
        """
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default="a cute fire pokemon with orange fur and flames",
        help='Text description of the Pokemon to generate'
    )
    
    parser.add_argument(
        '--steps', '-s',
        type=int,
        default=25,
        help='Number of inference steps (15-50, higher = better quality)'
    )
    
    parser.add_argument(
        '--guidance', '-g',
        type=float,
        default=7.5,
        help='Guidance scale (5.0-15.0, higher = more prompt adherence)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        choices=[256, 512, 768],
        help='Image size (256/512/768)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (auto/cuda/cpu)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Setup
    print("🚀 Pokemon LoRA Generator")
    print("=" * 50)
    
    setup_environment()
    check_system()
    
    # Download/check model
    model_path = download_model()
    
    # Generate image
    image = generate_pokemon(args.prompt, model_path, args)
    
    if image is None:
        print("❌ Generation failed!")
        sys.exit(1)
    
    # Save image
    if args.output:
        output_path = project_root / 'outputs' / args.output
    else:
        # Auto-generate filename
        safe_prompt = "".join(c for c in args.prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        output_path = project_root / 'outputs' / f"pokemon_{safe_prompt}_{args.seed}.png"
    
    image.save(output_path)
    print(f"💾 Image saved: {output_path}")
    
    # Show image info
    print(f"📊 Image info:")
    print(f"   Size: {image.size}")
    print(f"   Prompt: '{args.prompt}'")
    print(f"   Steps: {args.steps}")
    print(f"   Seed: {args.seed}")
    
    print("\n🎉 Generation complete!")
    print(f"💡 Try different prompts or seeds for variety!")
    print(f"🎨 Your Pokemon is ready: {output_path}")

if __name__ == "__main__":
    main()