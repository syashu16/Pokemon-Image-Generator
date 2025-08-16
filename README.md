# 🎨 Pokemon LoRA - AI Pokemon Generator

Generate your own Pokemon-style characters using AI! This project uses a fine-tuned Stable Diffusion model with LoRA (Low-Rank Adaptation) to create unique Pokemon-inspired artwork.

## ✨ Features

- 🎯 **Easy to Use**: One-click setup and generation
- 🚀 **Fast**: Generates images in 10-20 seconds
- 💾 **Memory Efficient**: Works on 4GB GPU (GTX 1650 Ti and above)
- 🎨 **High Quality**: Fine-tuned specifically for Pokemon-style art
- 🔧 **Beginner Friendly**: No AI/ML knowledge required
- 🤗 **Hugging Face Integration**: Available on Hugging Face Hub

## 🤗 Hugging Face Model

**Model**: [`yashu16/pokemon-lora-v1`](https://huggingface.co/yashu16/pokemon-lora-v1)
- **Base Model**: Stable Diffusion v1.5
- **Method**: LoRA fine-tuning
- **Hardware**: Trained on GTX 1650 Ti (4GB VRAM)
- **License**: MIT
- **Created**: August 16, 2025

### Quick Usage with Diffusers

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.load_lora_weights("yashu16/pokemon-lora-v1")

image = pipeline("a cute fire pokemon with orange fur").images[0]
image.save("my_pokemon.png")
```

## 🚀 Quick Start (Beginners)

### Option 1: One-Click Setup (Recommended)

1. **Download this repository**
   ```bash
   git clone https://github.com/syashu16/pokemon-lora-generator.git
   cd pokemon-lora-generator
   ```

2. **Run the setup script**
   ```bash
   # Windows
   setup.bat
   
   # Mac/Linux
   bash setup.sh
   ```

3. **Generate your first Pokemon!**
   ```bash
   python generate.py
   ```

### Option 2: Use Hugging Face Diffusers Directly

1. **Install dependencies**
   ```bash
   pip install diffusers transformers torch
   ```

2. **Use the model**
   ```python
   from diffusers import StableDiffusionPipeline
   import torch
   
   # Load the pipeline
   pipeline = StableDiffusionPipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5",
       torch_dtype=torch.float16
   )
   
   # Load your LoRA weights
   pipeline.load_lora_weights("yashu16/pokemon-lora-v1")
   
   # Generate Pokemon
   prompt = "a cute fire pokemon with orange fur"
   image = pipeline(prompt).images[0]
   image.save("pokemon.png")
   ```

### Option 3: Manual Setup

1. **Install Python 3.8+** from [python.org](https://python.org)

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the generator**
   ```bash
   python generate.py
   ```

## 🎮 How to Use

### Simple Generation
```bash
python generate.py --prompt "a cute fire pokemon with orange fur"
```

### Advanced Options
```bash
python generate.py \
  --prompt "a water pokemon swimming with blue scales" \
  --steps 25 \
  --guidance 7.5 \
  --seed 42 \
  --output my_pokemon.png
```

### Interactive Mode
```bash
python interactive.py
```

## 📝 Prompt Examples

### 🔥 Fire Type
- `"a cute fire pokemon with orange fur and small flames"`
- `"a dragon fire pokemon with red scales breathing flames"`
- `"a small fire pokemon with big eyes and warm glow"`

### 💧 Water Type  
- `"a blue water pokemon with fins swimming gracefully"`
- `"a large water pokemon with shell and ocean waves"`
- `"a cute water pokemon with bubbles and aqua colors"`

### ⚡ Electric Type
- `"a yellow electric pokemon with lightning bolt patterns"`
- `"a small electric pokemon with sparks and bright energy"`
- `"a powerful electric pokemon with glowing electric aura"`

### 🌿 Grass Type
- `"a green grass pokemon covered in leaves and flowers"`
- `"a forest grass pokemon with tree branches and vines"`
- `"a small grass pokemon with petals and nature elements"`

## 🛠️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **RAM**: 8GB
- **Storage**: 10GB free space
- **GPU**: GTX 1650 Ti (4GB VRAM) or better
- **Python**: 3.8 or higher

### Recommended Requirements
- **GPU**: RTX 3060 (8GB VRAM) or better
- **RAM**: 16GB
- **Storage**: 20GB free space (SSD preferred)

### CPU-Only Mode
- Works without GPU but slower (2-5 minutes per image)
- Requires 16GB RAM for best performance

## 📁 Project Structure

```
pokemon-lora-generator/
├── 📄 README.md              # This file
├── 🚀 generate.py             # Main generation script
├── 🎮 interactive.py          # Interactive web interface
├── 🔧 setup.py               # Installation script
├── 📋 requirements.txt       # Python dependencies
├── 🏗️ models/                # Model files (auto-downloaded)
├── 🖼️ outputs/               # Generated images
├── 📚 examples/              # Example prompts and outputs
└── 🛠️ utils/                 # Utility functions
```

## 🔧 Advanced Usage

### Using the Hugging Face Model Directly

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# Initialize pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

# Load LoRA weights
pipe.load_lora_weights("yashu16/pokemon-lora-v1")

# Optional: Use faster scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Move to GPU if available
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# Generate image
prompt = "a majestic electric pokemon with golden lightning patterns"
image = pipe(
    prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

image.save("electric_pokemon.png")
```

### Training Your Own LoRA

Want to train on your own art style? Use our training script:

```bash
python train_custom_lora.py --dataset_path "path/to/your/images"
```

### Batch Generation

Generate multiple images at once:

```bash
python batch_generate.py --prompts_file prompts.txt --count 10
```

### API Usage

Use as a Python library:

```python
from pokemon_lora import PokemonGenerator

generator = PokemonGenerator()
image = generator.generate("a cute electric pokemon")
image.save("my_pokemon.png")
```

## 🐛 Troubleshooting

### Common Issues

**❌ "CUDA out of memory"**
```bash
# Solution: Use CPU mode or reduce image size
python generate.py --device cpu --size 256
```

**❌ "Model not found"**
```bash
# Solution: Re-download models
python setup.py --redownload
```

**❌ "Import error"**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**❌ "LoRA weights not loading"**
```bash
# Solution: Ensure you have the latest diffusers version
pip install --upgrade diffusers
```

### Performance Tips

- **Faster Generation**: Use `--steps 15` for quicker results
- **Better Quality**: Use `--steps 50` for higher quality
- **Memory Saving**: Use `--size 256` for smaller images
- **Batch Processing**: Generate multiple images with `batch_generate.py`

### Development Setup

```bash
git clone https://github.com/syashu16/pokemon-lora-generator.git
cd pokemon-lora-generator
pip install -r requirements-dev.txt
python -m pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stable Diffusion**: Stability AI for the base model
- **LoRA**: Microsoft for Low-Rank Adaptation technique
- **Pokemon Dataset**: Community-contributed Pokemon artwork
- **Diffusers**: Hugging Face for the excellent library
- **Hugging Face**: For hosting our model at [`yashu16/pokemon-lora-v1`](https://huggingface.co/yashu16/pokemon-lora-v1)

**Made with ❤️ by Yashu and Varun

*Generate amazing Pokemon-style art with AI! ⭐ Star this repo if you found it useful!*

**🤗 Available on Hugging Face: [`yashu16/pokemon-lora-v1`](https://huggingface.co/yashu16/pokemon-lora-v1)**
