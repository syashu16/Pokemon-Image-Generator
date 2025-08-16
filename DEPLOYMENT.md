# 🚀 Deployment Guide - Pokemon LoRA Generator

This guide covers different ways to deploy and share your Pokemon LoRA Generator.

## 🌐 Web Deployment Options

### 1. Streamlit Cloud (Recommended for Beginners)

**Pros:** Free, easy setup, automatic updates
**Cons:** Limited resources, CPU-only

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy `streamlit_app.py`

**Configuration:**
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### 2. Hugging Face Spaces

**Pros:** Free GPU access, ML-focused community
**Cons:** Queue system during high usage

**Steps:**
1. Create account on [huggingface.co](https://huggingface.co)
2. Create new Space with Streamlit
3. Upload your files
4. Add `requirements.txt`

**Space Configuration:**
```yaml
# README.md header
---
title: Pokemon LoRA Generator
emoji: 🎨
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: streamlit_app.py
pinned: false
---
```

### 3. Google Colab (Free GPU)

**Pros:** Free GPU access, Jupyter environment
**Cons:** Session timeouts, not permanent

**Colab Notebook:**
```python
# Install dependencies
!pip install streamlit diffusers transformers torch

# Clone repository
!git clone https://github.com/syashu16/pokemon-lora-generator.git
%cd pokemon-lora-generator

# Run Streamlit with ngrok
!pip install pyngrok
from pyngrok import ngrok
import subprocess
import threading

# Start Streamlit in background
def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit_app.py", "--server.port", "8501"])

thread = threading.Thread(target=run_streamlit)
thread.start()

# Create public URL
public_url = ngrok.connect(8501)
print(f"🌐 Access your app at: {public_url}")
```

### 4. Railway (Paid but Simple)

**Pros:** Easy deployment, persistent storage
**Cons:** Paid service

**railway.json:**
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0"
  }
}
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p models/cache outputs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  pokemon-lora:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Build and Run
```bash
# Build image
docker build -t pokemon-lora-generator .

# Run container
docker run -p 8501:8501 pokemon-lora-generator

# With GPU support
docker run --gpus all -p 8501:8501 pokemon-lora-generator
```

## ☁️ Cloud Platforms

### AWS EC2

**Instance Recommendations:**
- **CPU-only**: t3.large (2 vCPU, 8GB RAM)
- **GPU**: g4dn.xlarge (4 vCPU, 16GB RAM, T4 GPU)

**Setup Script:**
```bash
#!/bin/bash
# EC2 User Data Script

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and Git
sudo apt install -y python3 python3-pip git

# Clone repository
git clone https://github.com/syashu16/pokemon-lora-generator.git
cd pokemon-lora-generator

# Install dependencies
pip3 install -r requirements.txt

# Install NVIDIA drivers (for GPU instances)
sudo apt install -y nvidia-driver-470

# Start application
nohup streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
```

### Google Cloud Platform

**Compute Engine Setup:**
```bash
# Create instance with GPU
gcloud compute instances create pokemon-lora-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB \
    --metadata-from-file startup-script=startup.sh
```

### Azure

**Container Instance:**
```yaml
# azure-container.yml
apiVersion: 2019-12-01
location: eastus
name: pokemon-lora-generator
properties:
  containers:
  - name: pokemon-lora
    properties:
      image: your-registry/pokemon-lora-generator:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 8
      ports:
      - port: 8501
        protocol: TCP
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8501
```

## 📱 Mobile-Friendly Deployment

### Progressive Web App (PWA)

Add to your Streamlit app:

```python
# streamlit_app.py
import streamlit as st

# PWA configuration
st.set_page_config(
    page_title="Pokemon LoRA Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add PWA manifest
st.markdown("""
<link rel="manifest" href="data:application/json;base64,ewogICJuYW1lIjogIlBva2Vtb24gTG9SQSBHZW5lcmF0b3IiLAogICJzaG9ydF9uYW1lIjogIlBva2Vtb24gQUkiLAogICJzdGFydF91cmwiOiAiLyIsCiAgImRpc3BsYXkiOiAic3RhbmRhbG9uZSIsCiAgImJhY2tncm91bmRfY29sb3IiOiAiI0ZGRkZGRiIsCiAgInRoZW1lX2NvbG9yIjogIiNGRjZCNkIiLAogICJpY29ucyI6IFsKICAgIHsKICAgICAgInNyYyI6ICJkYXRhOmltYWdlL3N2Zyt4bWw7YmFzZTY0LC4uLiIsCiAgICAgICJzaXplcyI6ICIxOTJ4MTkyIiwKICAgICAgInR5cGUiOiAiaW1hZ2Uvc3ZnK3htbCIKICAgIH0KICBdCn0=">
""", unsafe_allow_html=True)
```

## 🔒 Security Considerations

### Environment Variables
```bash
# .env file (don't commit to git)
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key
SECRET_KEY=your_secret_key
```

### Rate Limiting
```python
# Add to streamlit_app.py
import time
from collections import defaultdict

# Simple rate limiting
user_requests = defaultdict(list)

def rate_limit_check(user_ip, max_requests=10, window=3600):
    now = time.time()
    user_requests[user_ip] = [req for req in user_requests[user_ip] if now - req < window]
    
    if len(user_requests[user_ip]) >= max_requests:
        return False
    
    user_requests[user_ip].append(now)
    return True
```

### Input Validation
```python
def validate_prompt(prompt):
    """Validate user input"""
    if len(prompt) > 500:
        return False, "Prompt too long"
    
    banned_words = ["inappropriate", "content"]
    if any(word in prompt.lower() for word in banned_words):
        return False, "Inappropriate content"
    
    return True, "Valid"
```

## 📊 Monitoring and Analytics

### Basic Logging
```python
import logging
import streamlit as st

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Log user interactions
def log_generation(prompt, success):
    logging.info(f"Generation: {prompt[:50]}... Success: {success}")
```

### Usage Analytics
```python
# Simple analytics
def track_usage():
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0
    
    st.session_state.usage_count += 1
    
    # Display in sidebar
    st.sidebar.metric("Generations Today", st.session_state.usage_count)
```

## 🚀 Performance Optimization

### Caching Strategies
```python
@st.cache_resource
def load_model():
    """Cache model loading"""
    return load_pokemon_model()

@st.cache_data
def process_prompt(prompt):
    """Cache prompt processing"""
    return preprocess_prompt(prompt)
```

### Memory Management
```python
import gc
import torch

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Call after each generation
cleanup_memory()
```

## 📈 Scaling Considerations

### Load Balancing
- Use nginx for multiple instances
- Implement queue system for GPU access
- Consider Redis for session management

### Database Integration
```python
# Optional: Store generations in database
import sqlite3

def save_generation(prompt, image_path, user_id):
    conn = sqlite3.connect('generations.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO generations (prompt, image_path, user_id, timestamp) VALUES (?, ?, ?, ?)",
        (prompt, image_path, user_id, datetime.now())
    )
    conn.commit()
    conn.close()
```

## 🎯 Deployment Checklist

- [ ] Code tested locally
- [ ] Requirements.txt updated
- [ ] Environment variables configured
- [ ] Security measures implemented
- [ ] Error handling added
- [ ] Logging configured
- [ ] Performance optimized
- [ ] Documentation updated
- [ ] Backup strategy planned
- [ ] Monitoring setup

## 🆘 Troubleshooting

### Common Issues

**Out of Memory:**
```python
# Reduce model precision
pipeline = pipeline.to(torch.float16)

# Enable memory optimizations
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()
```

**Slow Loading:**
```python
# Preload models
@st.cache_resource
def preload_models():
    return load_all_models()
```

**Connection Issues:**
```python
# Add retry logic
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(delay)
            return wrapper
        return decorator
```

---

**Need help with deployment?** Join our [Discord community](https://discord.gg/pokemon-lora) or open an issue on GitHub! 🚀