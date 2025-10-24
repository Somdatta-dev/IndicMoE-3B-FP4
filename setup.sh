#!/bin/bash
# IndicMoE-3B-FP4 Setup Script

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         IndicMoE-3B-FP4 Environment Setup                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Start Docker container
echo "🐳 Starting Docker container..."
docker compose up -d

# Wait for container to start
sleep 3

# Install Megatron-Core and dependencies
echo ""
echo "📦 Installing dependencies..."
docker exec indicmoe-training bash -c "
    # Show pre-installed versions
    echo '=== Pre-installed Components ===' && \
    python3 -c 'import torch; print(f\"✅ PyTorch: {torch.__version__}\")' && \
    python3 -c 'import transformer_engine as te; print(f\"✅ Transformer Engine: {te.__version__}\")' && \
    
    # Clone and install Megatron-LM
    echo '' && \
    echo '=== Installing Megatron-LM ===' && \
    cd /workspace && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    pip install -e . && \
    
    # Install additional requirements
    echo '' && \
    echo '=== Installing Python packages ===' && \
    cd /workspace && \
    pip install -r code/requirements.txt && \
    
    # Verify NVFP4 support
    echo '' && \
    echo '=== Verifying NVFP4 Support ===' && \
    python3 -c 'from megatron.core.transformer.transformer_config import TransformerConfig; import inspect; fp4_params = [p for p in inspect.signature(TransformerConfig).parameters if \"fp4\" in p]; print(f\"✅ FP4 parameters available: {fp4_params}\")' && \
    
    echo '' && \
    echo '✅ Installation complete!'
"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                  Setup Complete!                         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Next steps:"
echo "  1. Enter container:  docker exec -it indicmoe-training bash"
echo "  2. Test setup:       python3 code/train_moe_nvfp4.py"
echo "  3. Attach VS Code:   Remote Explorer → indicmoe-training"
echo ""
echo "📊 Monitoring:"
echo "  - TensorBoard:  http://localhost:6006"
echo "  - Jupyter Lab:  http://localhost:8888"
echo ""