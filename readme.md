# IndicMoE-3B-FP4

**Multilingual Indian Language Model with Mixture of Experts**

A 3 billion parameter language model trained with NVFP4 (4-bit precision) featuring 8 language-specific experts for English and major Indian languages.

## 🌟 Features

- **3B Parameters**: Efficient size for deployment while maintaining quality
- **8 Expert MoE Architecture**: Language-specific experts for optimal performance
- **NVFP4 Training**: 4-bit precision training on Blackwell architecture (RTX 5090)
- **Function Calling**: Built-in support for tool use and function calling
- **Multilingual**: English + 7 Indian languages

## 🗣️ Supported Languages

| Expert | Language | Script |
|--------|----------|--------|
| 0 | English | Latin |
| 1 | Hindi | देवनागरी |
| 2 | Tamil | தமிழ் |
| 3 | Telugu | తెలుగు |
| 4 | Bengali | বাংলা |
| 5 | Marathi | मराठी |
| 6 | Gujarati | ગુજરાતી |
| 7 | Kannada | ಕನ್ನಡ |

## 🏗️ Architecture

- **Model Type**: GPT-style decoder-only transformer
- **Total Parameters**: ~3B
- **Layers**: 32
- **Hidden Size**: 2560
- **Attention Heads**: 32
- **MoE Experts**: 8 (Top-2 routing)
- **Context Length**: 16000 tokens
- **Precision**: NVFP4 (4-bit training)

## 🚀 Quick Start

### Prerequisites

- Docker with GPU support (NVIDIA Container Toolkit)
- NVIDIA GPU with Blackwell architecture (RTX 5090 or better)
- 32GB+ VRAM
- WSL2 (for Windows users)

### Setup

```bash
# Clone the repository
git clone https://github.com/Somdatta-dev/IndicMoE-3B-FP4
cd indicmoe-3b-fp4

# Start Docker environment
docker compose up -d

# Run setup
./setup.sh

# Run setup Windows
./setup.bat

# Test configuration
docker exec -it indicmoe-training python3 code/train_moe_nvfp4.py
```

## 📊 Training Details

- **Framework**: NVIDIA Megatron-Core 0.16.0
- **Precision**: NVFP4 with Transformer Engine 2.7.0
- **Hardware**: NVIDIA RTX 5090 (32GB)
- **Batch Size**: 2 (with gradient accumulation)
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW

## 📁 Project Structure

```
indicmoe-3b-fp4/
├── code/
│   ├── train_moe_nvfp4.py     # Training script
│   └── requirements.txt       # Python dependencies
├── configs/                   # Config files
├── docker-compose.yml         # Docker configuration
├── setup.sh                   # Setup script
└── README.md
```

## 🔬 Technical Innovation

NVFP4 Training: This model is trained using NVIDIA's NVFP4 (4-bit floating point) format, enabling:

- Reduced memory footprint
- Faster training on Blackwell GPUs
- Maintained model quality through block scaling and Hadamard transforms

## 📝 Citation

```bibtex
@software{indicmoe2025,
  title={IndicMoE-3B-FP4: Multilingual Indian Language Model},
  author={Somdatta Chakravarty},
  year={2025},
  url={https://github.com/Somdatta-dev/IndicMoE-3B-FP4}
}
```

## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- NVIDIA for Megatron-Core and Transformer Engine
- AI4Bharat for Indian language resources
- Research paper: NVFP4 Training Paper