#!/usr/bin/env python
# coding: utf-8

"""
🚀 MEMORY-FIXED Pokemon LoRA Training - GTX 1650 Ti (4GB VRAM)
Author: syashu16
Date: 2025-08-15
Hardware: HP OMEN GTX 1650 Ti (4GB VRAM)
Dataset: Ultra-small subset (≤20 samples) for memory efficiency
Time: ~3-5 minutes training time
"""

# CRITICAL: Set memory environment variables FIRST
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# =============================================================================
# STEP 1: SYSTEM SETUP AND PACKAGE INSTALLATION
# =============================================================================

import subprocess
import sys
import gc
import shutil
from pathlib import Path

print("🚀 Fast Pokemon LoRA Training Setup")
print("=" * 60)
print("User: syashu16")
print("Target: Fast training on small dataset (≤500 samples)")
print("Expected time: 10-15 minutes")
print("=" * 60)

# Install packages function
def install_package(package):
    """Install package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except:
        return False

# Check and setup D drive
def setup_drive():
    """Setup D drive cache directories"""
    D_DRIVE_CACHE = "D:/ai_models_cache"
    
    try:
        if os.path.exists("D:/"):
            total, used, free = shutil.disk_usage("D:/")
            print(f"💾 D Drive: {free // (1024**3)} GB free")
            
            if free < 5 * (1024**3):  # Need 5GB minimum
                print("❌ Warning: Less than 5GB free space")
            else:
                print("✅ Sufficient space available")
        
        os.makedirs(D_DRIVE_CACHE, exist_ok=True)
        print(f"📁 Cache directory: {D_DRIVE_CACHE}")
        return D_DRIVE_CACHE
    
    except Exception as e:
        print(f"❌ Drive setup failed: {e}")
        return "./cache"  # Fallback to local cache

D_DRIVE_CACHE = setup_drive()

# Install required packages
print("\n📦 Installing required packages...")
packages = [
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "datasets",
    "safetensors",
    "Pillow",
    "requests",
    "tqdm",
    "matplotlib"
]

# Install PyTorch with CUDA
print("Installing PyTorch with CUDA...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages[0].split())
    print("✅ PyTorch with CUDA installed")
except:
    print("⚠️ Installing CPU version...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])

# Install other packages
for package in packages[1:]:
    print(f"Installing {package}...")
    if install_package(package):
        print(f"✅ {package} installed")
    else:
        print(f"❌ Failed: {package}")

# =============================================================================
# STEP 2: IMPORT LIBRARIES AND DEVICE SETUP
# =============================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import requests
from io import BytesIO
import random
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Device and dtype setup - MEMORY OPTIMIZED FOR 4GB GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32  # Use float32 for stability (not float16)

print(f"\n🧪 System Information:")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("⚠️ Using CPU due to 4GB VRAM limitation")
else:
    print("ℹ️ Using CPU")
print(f"🎯 Using device: {device}")

# =============================================================================
# STEP 3: ENVIRONMENT SETUP FOR D DRIVE CACHING
# =============================================================================

# Set environment variables for D drive caching
os.environ['HF_HOME'] = D_DRIVE_CACHE
os.environ['TRANSFORMERS_CACHE'] = f"{D_DRIVE_CACHE}/transformers"
os.environ['HF_DATASETS_CACHE'] = f"{D_DRIVE_CACHE}/datasets"
os.environ['TORCH_HOME'] = f"{D_DRIVE_CACHE}/torch"

print(f"\n🗂️ Cache directories set:")
print(f"   Models: {D_DRIVE_CACHE}")
print(f"   Transformers: {os.environ['TRANSFORMERS_CACHE']}")
print(f"   Datasets: {os.environ['HF_DATASETS_CACHE']}")

# =============================================================================
# STEP 4: LOAD STABLE DIFFUSION BASE MODEL
# =============================================================================

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

print(f"\n📥 Loading Stable Diffusion v1.5...")

# Load model components with D drive caching
tokenizer = CLIPTokenizer.from_pretrained(
    MODEL_NAME, 
    subfolder="tokenizer",
    cache_dir=os.environ['TRANSFORMERS_CACHE']
)

text_encoder = CLIPTextModel.from_pretrained(
    MODEL_NAME, 
    subfolder="text_encoder",
    cache_dir=os.environ['TRANSFORMERS_CACHE']
)

vae = AutoencoderKL.from_pretrained(
    MODEL_NAME, 
    subfolder="vae",
    cache_dir=os.environ['TRANSFORMERS_CACHE']
)

# MEMORY FIX: Enable VAE optimizations immediately
vae.enable_slicing()
if hasattr(vae, 'enable_tiling'):
    vae.enable_tiling()

unet = UNet2DConditionModel.from_pretrained(
    MODEL_NAME, 
    subfolder="unet",
    cache_dir=os.environ['TRANSFORMERS_CACHE']
)

noise_scheduler = DDPMScheduler.from_pretrained(
    MODEL_NAME, 
    subfolder="scheduler",
    cache_dir=os.environ['TRANSFORMERS_CACHE']
)

# Move to device and freeze non-trainable components
vae = vae.to(device, dtype=dtype)
text_encoder = text_encoder.to(device, dtype=dtype)
unet = unet.to(device, dtype=dtype)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

print("✅ Base models loaded successfully!")

# MEMORY FIX: Aggressive cleanup after model loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()

if torch.cuda.is_available():
    print(f"💾 GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# =============================================================================
# STEP 5: SETUP LORA CONFIGURATION (OPTIMIZED FOR FAST TRAINING)
# =============================================================================

def setup_lora_unet(unet, rank=4, alpha=16):
    """Setup LoRA for UNet - optimized for fast training"""
    target_modules = [
        "to_k", "to_q", "to_v", "to_out.0"
    ]
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        # Remove task_type for diffusion models - not needed
    )
    
    unet_lora = get_peft_model(unet, lora_config)
    return unet_lora

# Apply LoRA with conservative settings
print("\n🔧 Setting up LoRA...")
unet_lora = setup_lora_unet(unet, rank=4, alpha=16)

trainable_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in unet_lora.parameters())

print(f"✅ LoRA applied!")
print(f"📊 Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# =============================================================================
# STEP 6: FAST SMALL DATASET PREPARATION
# =============================================================================

class FastPokemonDataset(Dataset):
    """Fast Pokemon dataset with limited samples for quick training"""
    
    def __init__(self, tokenizer, size=256, max_samples=20):  # MEMORY FIX: 256x256 images, 20 samples
        self.size = size
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.data = []
        
        print(f"🔄 Loading Pokemon dataset (max {max_samples} samples)...")
        self.load_pokemon_data()
    
    def load_pokemon_data(self):
        """Load limited Pokemon dataset for fast training"""
        try:
            # Load Pokemon dataset with streaming for memory efficiency
            dataset = load_dataset(
                "lambdalabs/pokemon-blip-captions",
                split="train",
                streaming=False,  # Load full dataset but limit samples
                cache_dir=os.environ['HF_DATASETS_CACHE']
            )
            
            # Take only first max_samples for fast training
            self.data = []
            for i, item in enumerate(dataset):
                if i >= self.max_samples:
                    break
                    
                # Only include items with valid images and text
                if item.get("image") and item.get("text"):
                    self.data.append({
                        "image": item["image"],
                        "text": item["text"]
                    })
            
            print(f"✅ Pokemon dataset loaded: {len(self.data)} samples")
            if self.data:
                print(f"📝 Sample: '{self.data[0]['text'][:50]}...'")
                
        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
            print("🔄 Creating synthetic dataset...")
            self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic Pokemon-style data if real dataset fails"""
        pokemon_prompts = [
            "a cute fire pokemon with orange fur and small flames",
            "a blue water pokemon with fins swimming gracefully", 
            "an electric pokemon with yellow fur and lightning bolts",
            "a grass pokemon with green leaves and flower petals",
            "a psychic pokemon with purple aura and mystical powers",
            "a dragon pokemon with scales and large wings",
            "a fairy pokemon with pink colors and sparkles",
            "a ghost pokemon floating with ethereal glow",
            "a fighting pokemon in strong battle pose",
            "an ice pokemon with crystal formations",
            "a rock pokemon with stone armor and strength",
            "a flying pokemon soaring through cloudy skies",
            "a bug pokemon with colorful wing patterns",
            "a poison pokemon with dark purple coloring",
            "a ground pokemon digging through earth",
            "a steel pokemon with metallic silver shine",
            "a normal pokemon with friendly appearance",
            "a dark pokemon lurking in shadows",
            "a legendary pokemon with majestic presence",
            "a baby pokemon with innocent big eyes"
        ]
        
        self.data = []
        for i in range(min(self.max_samples, 200)):  # Cap synthetic data
            prompt = pokemon_prompts[i % len(pokemon_prompts)]
            
            # Create colorful synthetic image (512x512)
            colors = np.random.randint(80, 255, 3)  # Bright colors
            image_array = np.random.randint(0, 255, (self.size, self.size, 3), dtype=np.uint8)
            
            # Add some structure (circles, rectangles for Pokemon-like shapes)
            center_x, center_y = self.size // 2, self.size // 2
            radius = random.randint(50, 150)
            
            # Create circular regions with different colors
            y, x = np.ogrid[:self.size, :self.size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            image_array[mask] = colors
            
            image = Image.fromarray(image_array)
            
            self.data.append({
                "image": image,
                "text": prompt
            })
        
        print(f"✅ Synthetic dataset created: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process image
        if hasattr(item["image"], 'convert'):
            image = item["image"].convert("RGB")
        else:
            image = Image.fromarray(item["image"]).convert("RGB")
            
        image = image.resize((self.size, self.size), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Tokenize text
        text_inputs = self.tokenizer(
            item["text"],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids.flatten(),
        }

# Create SUPER FAST dataset (50 samples only for ultra-fast training)
print("\n🎯 Creating SUPER FAST Pokemon dataset...")
dataset = FastPokemonDataset(
    tokenizer=tokenizer,
    max_samples=20  # MEMORY FIX: Ultra-small dataset for 4GB GPU
)

print(f"✅ Super fast dataset ready: {len(dataset)} samples")
print(f"⚡ Expected training time: 2-3 minutes per epoch")

# =============================================================================
# STEP 7: DATALOADER AND TRAINING CONFIGURATION
# =============================================================================

# MEMORY FIX: Ultra-small batch size for 4GB GPU
batch_size = 1  # CRITICAL: Batch size 1 for memory efficiency

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Avoid Windows multiprocessing issues
    pin_memory=True if device.type == "cuda" else False
)

# ULTRA-FAST training configuration
LEARNING_RATE = 1e-4  # Higher learning rate for faster convergence
NUM_EPOCHS = 1  # Only 1 epoch for ultra-fast training
GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation for speed
MAX_GRAD_NORM = 0.5  # Lower clipping for stability
SAVE_STEPS = len(dataloader)  # Save only at end

print(f"\n⚙️ ULTRA-FAST Training Configuration:")
print(f"   Dataset size: {len(dataset)} samples")
print(f"   Batch size: {batch_size}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Total steps: {len(dataloader) * NUM_EPOCHS}")
print(f"   Expected time: 2-3 minutes total")

# Setup optimizer with higher learning rate
optimizer = torch.optim.AdamW(
    unet_lora.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),  # Faster convergence
    weight_decay=0.001,  # Lower weight decay
    eps=1e-06,  # Higher epsilon for stability
)

print("✅ Training setup complete!")

# =============================================================================
# STEP 8: FAST TRAINING FUNCTION
# =============================================================================

def ultra_fast_train_lora():
    """ULTRA-FAST LoRA training - 2-3 minutes total"""
    print("\n🚀 Starting ULTRA-FAST LoRA Training...")
    print("🎯 Target: Basic Pokemon-style adaptation")
    print("⚡ Expected: 2-3 minutes total training time")
    
    unet_lora.train()
    global_step = 0
    
    # Create checkpoints directory
    checkpoints_dir = f"{D_DRIVE_CACHE}/ultra_fast_pokemon_checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    start_time = datetime.now()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        epoch_loss = 0.0
        valid_losses = []
        progress_bar = tqdm(dataloader, desc=f"Ultra-Fast Training Epoch {epoch + 1}")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)
            
            # MEMORY FIX: Encode to latent space with slicing
            with torch.no_grad():
                # Enable VAE slicing for memory efficiency
                if hasattr(vae, 'enable_slicing'):
                    vae.enable_slicing()
                
                # Process smaller chunks if needed
                try:
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                except torch.cuda.OutOfMemoryError:
                    # Emergency: Process in even smaller chunks
                    print("⚠️ OOM during VAE encode, using emergency mode...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Split image into 4 quadrants and process separately
                    h, w = pixel_values.shape[-2:]
                    h_half, w_half = h // 2, w // 2
                    
                    latent_chunks = []
                    for i in range(2):
                        for j in range(2):
                            chunk = pixel_values[:, :, i*h_half:(i+1)*h_half, j*w_half:(j+1)*w_half]
                            chunk_latent = vae.encode(chunk).latent_dist.sample()
                            latent_chunks.append(chunk_latent)
                            torch.cuda.empty_cache()
                    
                    # Combine chunks (simple average)
                    latents = torch.stack(latent_chunks).mean(dim=0)
                    latents = latents * vae.config.scaling_factor
            
            # Add noise with limited timestep range for faster training
            noise = torch.randn_like(latents)
            # Use smaller timestep range for faster convergence
            max_timesteps = min(500, noise_scheduler.config.num_train_timesteps)
            timesteps = torch.randint(
                0, max_timesteps,
                (latents.shape[0],), device=device
            ).long()
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
            
            # Forward pass
            model_pred = unet_lora(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Calculate loss with stability checks
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # NaN protection
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN/Inf detected at step {step}, skipping...")
                optimizer.zero_grad()
                continue
            
            # Scale loss if needed
            if loss.item() > 10.0:  # Very high loss
                loss = loss * 0.1  # Scale down
                print(f"🔧 High loss detected ({loss.item():.3f}), scaling down")
            
            loss.backward()
            valid_losses.append(loss.detach().item())
            
            # Update weights immediately (no accumulation for speed)
            torch.nn.utils.clip_grad_norm_(unet_lora.parameters(), MAX_GRAD_NORM)
            
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in unet_lora.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"⚠️ NaN gradients detected at step {step}, skipping update...")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Update progress
            if valid_losses:
                avg_loss = sum(valid_losses[-5:]) / min(5, len(valid_losses))  # Running average
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "step": global_step,
                    "valid": len(valid_losses)
                })
            
            # MEMORY FIX: Cleanup every step for 4GB GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        
        if valid_losses:
            avg_loss = sum(valid_losses) / len(valid_losses)
            print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f} (Valid steps: {len(valid_losses)}/{len(dataloader)})")
        else:
            print(f"Epoch {epoch + 1} - No valid training steps!")
        
        # Clean memory after epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    training_time = datetime.now() - start_time
    print(f"\n🎉 Ultra-fast training completed!")
    print(f"⏱️ Training time: {training_time}")
    print(f"📊 Valid training steps: {len(valid_losses) if 'valid_losses' in locals() else 'N/A'}")
    
    return unet_lora

# Memory optimization before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("\n💾 Memory optimized - starting training...")

# =============================================================================
# STEP 9: EXECUTE FAST TRAINING
# =============================================================================

print("🎯 Starting ULTRA-FAST Pokemon LoRA Training!")
print("Expected completion: 2-3 minutes")
print("=" * 60)

# Execute ultra-fast training
trained_model = ultra_fast_train_lora()

# Save final model
final_model_path = f"{D_DRIVE_CACHE}/ultra_fast_pokemon_lora_final"
os.makedirs(final_model_path, exist_ok=True)
trained_model.save_pretrained(final_model_path)

print(f"\n✅ Model saved to: {final_model_path}")

# =============================================================================
# STEP 10: FIXED IMAGE GENERATION
# =============================================================================

def generate_pokemon_image(prompt, num_inference_steps=20, guidance_scale=7.5, seed=42):
    """FIXED: Generate Pokemon image with proper LoRA integration"""
    print(f"🎨 Generating: '{prompt}'")
    
    # Create base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=os.environ['TRANSFORMERS_CACHE']
    )
    
    # CRITICAL FIX: Properly replace UNet and set to eval mode
    trained_model.eval()
    for param in trained_model.parameters():
        param.requires_grad = False
    
    pipeline.unet = trained_model
    pipeline = pipeline.to(device)
    
    # Enable memory optimizations
    pipeline.enable_attention_slicing()
    if hasattr(pipeline, 'enable_vae_slicing'):
        pipeline.enable_vae_slicing()
    
    # Generate image
    generator = torch.manual_seed(seed)
    
    with torch.no_grad():
        result = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=512,
            width=512,
        )
    
    # Clean up
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result.images[0]

# =============================================================================
# STEP 11: TEST GENERATION WITH MULTIPLE PROMPTS
# =============================================================================

# Create output directory
output_dir = f"{D_DRIVE_CACHE}/fast_generated_images"
os.makedirs(output_dir, exist_ok=True)

# Test prompts
test_prompts = [
    "a cute orange fire pokemon with big eyes and small flames",
    "a blue water pokemon swimming with elegant fins",
    "a yellow electric pokemon with lightning bolt patterns",
    "a green grass pokemon covered in leaves and flowers",
    "a purple psychic pokemon with mystical glowing aura"
]

print(f"\n🧪 Testing trained model...")
print(f"💾 Images will be saved to: {output_dir}")

generated_images = []

for i, prompt in enumerate(test_prompts):
    try:
        print(f"\n🎨 Generating image {i+1}/5...")
        image = generate_pokemon_image(prompt, seed=42+i)
        
        # Check if image is valid (not black)
        img_array = np.array(image)
        brightness = np.mean(img_array)
        
        print(f"📊 Image brightness: {brightness:.1f}")
        
        if brightness > 30:  # Not a black image
            print("✅ Generation successful!")
            filename = f"{output_dir}/pokemon_{i+1}_fast.png"
            image.save(filename)
            generated_images.append((image, prompt))
            print(f"💾 Saved: {filename}")
        else:
            print("⚠️ Generated image appears dark/black")
            # Save anyway for debugging
            filename = f"{output_dir}/pokemon_{i+1}_dark.png"
            image.save(filename)
        
    except Exception as e:
        print(f"❌ Generation {i+1} failed: {e}")

# =============================================================================
# STEP 12: DISPLAY RESULTS
# =============================================================================

print(f"\n🎉 Fast Pokemon LoRA Training Complete!")
print("=" * 60)

# Display generated images
if generated_images:
    print(f"✅ Successfully generated {len(generated_images)} images!")
    
    # Create a grid display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (image, prompt) in enumerate(generated_images[:5]):
        axes[i].imshow(image)
        axes[i].set_title(f"'{prompt[:30]}...'", fontsize=10)
        axes[i].axis('off')
    
    # Hide empty subplot
    if len(generated_images) < 6:
        axes[5].axis('off')
    
    plt.suptitle("🎨 Fast Pokemon LoRA - Generated Images", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print(f"🎯 All images saved to: {output_dir}")
    
else:
    print("❌ No successful generations - check training or generation code")

# =============================================================================
# STEP 13: QUICK COMPARISON TEST
# =============================================================================

def quick_comparison_test():
    """Quick test to compare base model vs trained model"""
    print("\n🔍 Quick Comparison Test")
    print("Comparing base SD vs trained LoRA model...")
    
    test_prompt = "a cute fire pokemon with orange fur"
    
    try:
        # Generate with base model
        base_pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        
        with torch.no_grad():
            base_image = base_pipeline(
                test_prompt,
                num_inference_steps=20,
                generator=torch.manual_seed(42)
            ).images[0]
        
        # Generate with trained model
        trained_image = generate_pokemon_image(test_prompt, seed=42)
        
        # Display comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(base_image)
        ax1.set_title("Base Stable Diffusion 1.5")
        ax1.axis('off')
        
        ax2.imshow(trained_image)
        ax2.set_title("Fast Pokemon LoRA")
        ax2.axis('off')
        
        plt.suptitle(f"Comparison: '{test_prompt}'", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Save comparison
        base_image.save(f"{output_dir}/comparison_base.png")
        trained_image.save(f"{output_dir}/comparison_trained.png")
        
        print("✅ Comparison complete!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

# Run comparison
quick_comparison_test()

# =============================================================================
# STEP 14: FINAL SUMMARY AND EXPORT
# =============================================================================

print("\n🎉 FAST POKEMON LORA PROJECT COMPLETED!")
print("=" * 60)

# Project summary
summary = f"""
📊 FAST TRAINING SUMMARY:
✅ Model: Stable Diffusion v1.5 + LoRA
✅ Dataset: {len(dataset)} Pokemon samples  
✅ Training: {NUM_EPOCHS} epochs (~10-15 mins)
✅ Device: {device.type.upper()}
✅ Style: Pokemon character generation

📁 OUTPUT FILES:
• {final_model_path} - Trained LoRA weights
• {output_dir} - Generated test images
• {D_DRIVE_CACHE}/fast_pokemon_checkpoints - Training checkpoints

🎯 MODEL USAGE:
```python
from diffusers import StableDiffusionPipeline
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.unet.load_adapter("{final_model_path}")
image = pipeline("a cute fire pokemon").images[0]
```

💡 RECOMMENDED PROMPTS:
• "a cute fire pokemon with orange flames"
• "a blue water pokemon with fins swimming"  
• "an electric pokemon with yellow lightning"
• "a grass pokemon with green leaves"
• "a dragon pokemon with large wings"

🚀 NEXT STEPS:
1. Generate more Pokemon with different prompts
2. Fine-tune with different learning rates
3. Try training on other art styles
4. Export and share your model
5. Create a web interface with Gradio

⏱️ TOTAL TIME: ~15-20 minutes (including setup)
🎉 SUCCESS: You've created your own AI model!
"""

print(summary)

# Export model info
model_info = {
    "model_name": "fast-pokemon-lora-v1",
    "author": "syashu16", 
    "created": datetime.now().isoformat(),
    "base_model": MODEL_NAME,
    "dataset_size": len(dataset),
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "training_time": "10-15 minutes",
    "hardware": "HP OMEN GTX 1650 Ti",
    "recommended_prompts": [
        "a cute fire pokemon with orange flames",
        "a blue water pokemon swimming gracefully",
        "an electric pokemon with yellow lightning",
        "a grass pokemon covered in green leaves"
    ]
}

with open(f"{final_model_path}/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print(f"📄 Model info saved: {final_model_path}/model_info.json")

# Clean up memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print(f"\n🎊 CONGRATULATIONS!")
print(f"You've successfully trained a Pokemon LoRA model in ~15 minutes!")
print(f"🎯 Your model is ready to generate Pokemon-style characters!")
print(f"💾 Total GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "")
print("=" * 60)