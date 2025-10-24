@echo off
REM IndicMoE-3B-FP4 Setup Script for Windows
chcp 65001 >nul

setlocal enabledelayedexpansion

cls
echo.
echo ============================================================
echo           IndicMoE-3B-FP4 Environment Setup
echo ============================================================
echo.

REM Start Docker container
echo [*] Starting Docker container...
docker compose up -d
if errorlevel 1 (
    echo Error: Failed to start Docker container
    pause
    exit /b 1
)

REM Wait for container to start
echo Waiting for container to initialize...
timeout /t 3 /nobreak

REM Install Megatron-Core and dependencies
echo.
echo [*] Installing dependencies...

docker exec indicmoe-training bash -c ^
    "echo '=== Pre-installed Components ===' && ^
    python3 -c 'import torch; print(f\"✅ PyTorch: {torch.__version__}\")' && ^
    python3 -c 'import transformer_engine as te; print(f\"✅ Transformer Engine: {te.__version__}\")' && ^
    echo '' && ^
    echo '=== Installing Megatron-LM ===' && ^
    cd /workspace && ^
    git clone https://github.com/NVIDIA/Megatron-LM.git && ^
    cd Megatron-LM && ^
    pip install -e . && ^
    echo '' && ^
    echo '=== Installing Python packages ===' && ^
    cd /workspace && ^
    pip install -r code/requirements.txt && ^
    echo '' && ^
    echo '=== Verifying NVFP4 Support ===' && ^
    python3 -c 'from megatron.core.transformer.transformer_config import TransformerConfig; import inspect; fp4_params = [p for p in inspect.signature(TransformerConfig).parameters if \"fp4\" in p]; print(f\"✅ FP4 parameters available: {fp4_params}\")' && ^
    echo '' && ^
    echo '✅ Installation complete!'"

if errorlevel 1 (
    echo Error: Failed to install dependencies in container
    pause
    exit /b 1
)

echo.
echo ============================================================
echo                   Setup Complete!
echo ============================================================
echo.
echo [*] Next steps:
echo   1. Enter container:  docker exec -it indicmoe-training bash
echo   2. Test setup:       python3 code/train_moe_nvfp4.py
echo   3. Attach VS Code:   Remote Explorer ^> indicmoe-training
echo.
echo [*] Monitoring:
echo   - TensorBoard:  http://localhost:6006
echo   - Jupyter Lab:  http://localhost:8888
echo.

pause
endlocal
