@echo off
echo 🚀 Pokemon LoRA Generator - Windows Setup
echo ========================================

echo.
echo 📋 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found!

echo.
echo 📦 Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies!
    echo 💡 Try: pip install --user -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!

echo.
echo 🔍 Checking for trained model...
if exist "D:\ai_models_cache\pokemon_lora_model" (
    echo ✅ Pokemon LoRA model found!
) else (
    echo ⚠️ Pokemon LoRA model not found.
    echo 💡 Please train your model first by running: python app.py
)

echo.
echo 🎉 Setup complete!
echo.
echo 🚀 To start the web interface, run:
echo    streamlit run streamlit_app.py
echo.
echo 🎮 To use command line, run:
echo    python generate.py --prompt "your pokemon description"
echo.
pause