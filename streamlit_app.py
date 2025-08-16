#!/usr/bin/env python3
"""
🎨 Working Pokemon LoRA Generator - Back to Basics
Simple version that actually works without errors
"""

import streamlit as st
import torch
import os
from pathlib import Path
import time
import random
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="🎨 Pokemon LoRA Generator",
    page_icon="🎨",
    layout="wide"
)

# Simple setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HOME'] = 'D:/ai_models_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/ai_models_cache/transformers'

@st.cache_resource
def load_working_model():
    """Load model the simple way that works"""
    try:
        from diffusers import StableDiffusionPipeline
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32  # Use float32 for stability
        
        st.info("📥 Loading Stable Diffusion model...")
        
        # Load base pipeline - simple approach
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir="D:/ai_models_cache/transformers"
        )
        
        # Try to load your LoRA
        lora_path = "D:/ai_models_cache/pokemon_lora_model"
        if Path(lora_path).exists():
            try:
                pipeline.load_lora_weights(lora_path)
                st.success("✅ Pokemon LoRA loaded!")
                has_lora = True
            except Exception as e:
                st.warning(f"⚠️ LoRA failed to load: {e}")
                st.info("Using base Stable Diffusion")
                has_lora = False
        else:
            st.warning("⚠️ No LoRA model found")
            st.info("Using base Stable Diffusion")
            has_lora = False
        
        # Simple setup
        pipeline = pipeline.to(device)
        pipeline.enable_attention_slicing()
        
        return pipeline, device, has_lora
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, False

def working_generate(pipeline, device, prompt, steps=20, guidance=7.5, seed=42, size=512):
    """Simple generation that works"""
    try:
        # Set models to eval mode
        pipeline.unet.eval()
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        with torch.no_grad():
            result = pipeline(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                height=size,
                width=size,
                negative_prompt="blurry, bad quality, ugly"
            )
            image = result.images[0]
        
        # Check brightness
        img_array = np.array(image)
        brightness = np.mean(img_array)
        
        return image, brightness, True
        
    except Exception as e:
        st.error(f"❌ Generation failed: {e}")
        return None, 0, False

def main():
    """Main app - keep it simple"""
    st.title("🎨 Pokemon LoRA Generator")
    st.markdown("### Simple and reliable Pokemon generation")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Info")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.success(f"🎯 GPU: {gpu_name}")
            st.info(f"💾 VRAM: {gpu_memory:.1f} GB")
        else:
            st.warning("⚠️ Using CPU")
    
    # Load model
    with st.spinner("📥 Loading model..."):
        pipeline, device, has_lora = load_working_model()
    
    if pipeline is None:
        st.error("❌ Could not load model!")
        st.stop()
    
    # Show model status
    with st.sidebar:
        st.header("🤖 Model Status")
        if has_lora:
            st.success("✅ Pokemon LoRA Active")
        else:
            st.warning("⚠️ Base Model Only")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎮 Generate Pokemon")
        
        # Prompt
        prompt = st.text_area(
            "📝 Describe your Pokemon:",
            value="a cute fire pokemon with orange fur and flames",
            height=100
        )
        
        # Quick buttons
        st.markdown("**🚀 Quick Prompts:**")
        cols = st.columns(3)
        
        quick_prompts = [
            ("🔥", "a cute fire pokemon with orange fur"),
            ("💧", "a blue water pokemon with fins"),
            ("⚡", "a yellow electric pokemon with lightning"),
            ("🌿", "a green grass pokemon with leaves"),
            ("🔮", "a purple psychic pokemon with aura"),
            ("🐉", "a dragon pokemon with wings")
        ]
        
        for i, (emoji, quick_prompt) in enumerate(quick_prompts):
            col_idx = i % 3
            if cols[col_idx].button(f"{emoji}"):
                prompt = quick_prompt
                st.rerun()
        
        # Settings
        st.markdown("---")
        st.markdown("**⚙️ Settings:**")
        
        steps = st.slider("Steps", 15, 30, 20, help="More steps = better quality")
        guidance = st.slider("Guidance", 5.0, 10.0, 7.5, help="Higher = follows prompt more")
        size = st.selectbox("Size", [256, 512], index=1, help="Larger = better quality")
        seed = st.number_input("Seed", 0, 999999, 42, help="Same seed = same result")
        
        # Generate button
        generate_btn = st.button("🎨 Generate Pokemon!", type="primary", use_container_width=True)
    
    with col2:
        st.header("🖼️ Generated Pokemon")
        
        if generate_btn:
            if not prompt.strip():
                st.error("❌ Please enter a prompt!")
            else:
                st.info(f"🎨 Generating: '{prompt}'")
                st.info(f"⚙️ Settings: {steps} steps, guidance {guidance}, seed {seed}")
                
                # Generate
                with st.spinner("🎨 Creating your Pokemon..."):
                    start_time = time.time()
                    image, brightness, success = working_generate(
                        pipeline, device, prompt, steps, guidance, seed, size
                    )
                    generation_time = time.time() - start_time
                
                if success and image:
                    st.image(image, caption=f"Generated: {prompt}", use_container_width=True)
                    
                    # Show results
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Generation Time", f"{generation_time:.1f}s")
                    with col_b:
                        st.metric("Brightness", f"{brightness:.1f}/255")
                    
                    # Status
                    if brightness > 30:
                        st.success("✅ Great image generated!")
                    elif brightness > 15:
                        st.warning("⚠️ Image a bit dark but okay")
                    else:
                        st.error("❌ Image too dark - try different prompt/seed")
                    
                    # Download
                    import io
                    buf = io.BytesIO()
                    image.save(buf, format='PNG')
                    st.download_button(
                        "💾 Download Image",
                        buf.getvalue(),
                        f"pokemon_{seed}.png",
                        "image/png",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Generation failed!")
        else:
            st.info("👆 Click 'Generate Pokemon!' to create your image")
            st.image("https://via.placeholder.com/400x400/FF6B6B/FFFFFF?text=Your+Pokemon+Here", 
                    caption="Your Pokemon will appear here")
    
    # Footer
    st.markdown("---")
    st.markdown("🎨 **Pokemon LoRA Generator** - Simple and reliable")

if __name__ == "__main__":
    import io
    main()