#!/bin/bash

echo "🚀 Pokemon LoRA Generator - Linux/Mac Setup"
echo "============================================"

# Check Python
echo ""
echo "📋 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi

python3 --version
echo "✅ Python found!"

# Check pip
echo ""
echo "📋 Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip not found! Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies!"
    echo "💡 Try: pip3 install --user -r requirements.txt"
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Check for model
echo ""
echo "🔍 Checking for trained model..."
if [ -d "D:/ai_models_cache/pokemon_lora_model" ] || [ -d "$HOME/.cache/pokemon_lora_model" ]; then
    echo "✅ Pokemon LoRA model found!"
else
    echo "⚠️ Pokemon LoRA model not found."
    echo "💡 Please train your model first by running: python3 app.py"
fi

# Make scripts executable
chmod +x generate.py
chmod +x run_web_app.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 To start the web interface, run:"
echo "   python3 run_web_app.py"
echo ""
echo "🎮 To use command line, run:"
echo "   python3 generate.py --prompt 'your pokemon description'"
echo ""