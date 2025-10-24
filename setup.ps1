# IndicMoE-3B-FP4 Setup Script for Windows (PowerShell)
# Run with: powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"

Clear-Host
Write-Host ""
Write-Host "============================================================"
Write-Host "           IndicMoE-3B-FP4 Environment Setup"
Write-Host "============================================================"
Write-Host ""

# Start Docker container
Write-Host "[*] Starting Docker container..."
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to start Docker container"
    Read-Host "Press Enter to exit"
    exit 1
}

# Wait for container to start
Write-Host "Waiting for container to initialize..."
Start-Sleep -Seconds 3

# Install Megatron-Core and dependencies
Write-Host ""
Write-Host "[*] Installing dependencies..."
Write-Host ""

$setupScript = @"
set -e

echo '=== Pre-installed Components ==='
python3 -c 'import torch; print(f"PyTorch: {torch.__version__}")'
python3 -c 'import transformer_engine as te; print(f"Transformer Engine: {te.__version__}")'

echo ''
echo '=== Installing Megatron-LM ==='
cd /workspace
if [ ! -d "Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git
fi
cd Megatron-LM
pip install -e .

echo ''
echo '=== Installing Python packages ==='
cd /workspace
pip install -r code/requirements.txt

echo ''
echo '=== Verifying NVFP4 Support ==='
python3 -c 'from megatron.core.transformer.transformer_config import TransformerConfig; import inspect; fp4_params = [p for p in inspect.signature(TransformerConfig).parameters if "fp4" in p]; print(f"FP4 parameters available: {fp4_params}")'

echo ''
echo '=== Installation complete! ==='
"@

docker exec indicmoe-training bash -c $setupScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies in container"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================"
Write-Host "                   Setup Complete!"
Write-Host "============================================================"
Write-Host ""
Write-Host "[*] Next steps:"
Write-Host "   1. Enter container:  docker exec -it indicmoe-training bash"
Write-Host "   2. Test setup:       python3 code/train_moe_nvfp4.py"
Write-Host "   3. Attach VS Code:   Remote Explorer > indicmoe-training"
Write-Host ""
Write-Host "[*] Monitoring:"
Write-Host "   - TensorBoard:  http://localhost:6006"
Write-Host "   - Jupyter Lab:  http://localhost:8888"
Write-Host ""

Read-Host "Press Enter to exit"
