"""
IndicMoE-3B-FP4 Training Script
"""
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel

from config import (
    MODEL_NAME, 
    MODEL_VERSION, 
    MODEL_FULL_NAME,
    LANGUAGES, 
    MODEL_CONFIG,
    TRAINING_CONFIG
)

def get_model_config():
    """Configure IndicMoE-3B-FP4"""
    return TransformerConfig(
        # Model Parallelism
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,

        # Model Architecture
        num_layers=MODEL_CONFIG["num_layers"],
        hidden_size=MODEL_CONFIG["hidden_size"],
        num_attention_heads=MODEL_CONFIG["num_attention_heads"],
        ffn_hidden_size=MODEL_CONFIG["ffn_hidden_size"],

        # MoE Configuration
        num_moe_experts=MODEL_CONFIG["num_moe_experts"],
        moe_router_topk=MODEL_CONFIG["moe_router_topk"],
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.01,

        # NVFP4 Configuration
        fp4=MODEL_CONFIG["fp4"],
        fp4_recipe=MODEL_CONFIG["fp4_recipe"],
        fp4_param=MODEL_CONFIG["fp4_param"],

        # Other settings
        attention_dropout=0.1,
        hidden_dropout=0.1,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )

def print_banner():
    """Print IndicMoE banner"""
    banner = f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              {MODEL_NAME}                      ║
    ║                                                           ║
    ║  Multilingual Indian Language Model                      ║
    ║  with Mixture of Experts                                 ║
    ║                                                           ║
    ║  Version: {MODEL_VERSION}                                      ║
    ║  Precision: NVFP4 (4-bit training)                       ║
    ║  Languages: {len(LANGUAGES)} (English + 7 Indian languages)       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    print_banner()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ No GPU available!")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize config
    config = get_model_config()
    
    print(f"\n{'='*60}")
    print(f"Architecture Configuration:")
    print(f"{'='*60}")
    print(f"Layers: {config.num_layers}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Attention Heads: {config.num_attention_heads}")
    print(f"FFN Size: {config.ffn_hidden_size}")
    
    print(f"\nMoE Configuration:")
    print(f"Experts: {config.num_moe_experts}")
    print(f"Router Top-K: {config.moe_router_topk}")
    
    print(f"\nLanguage Experts:")
    for expert_id, language in LANGUAGES.items():
        print(f"  Expert {expert_id}: {language.capitalize()}")
    
    print(f"\nNVFP4 Configuration:")
    print(f"FP4 Mode: {config.fp4}")
    print(f"FP4 Recipe: {config.fp4_recipe}")
    print(f"FP4 Param: {config.fp4_param}")
    print(f"{'='*60}\n")
    
    print("✅ Configuration ready!")
    print(f"\n🚀 {MODEL_NAME} is ready to train!")

if __name__ == "__main__":
    main()