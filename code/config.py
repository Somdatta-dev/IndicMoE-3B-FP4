"""
IndicMoE-3B-FP4 Configuration
"""

# Model Information
MODEL_NAME = "IndicMoE-3B-FP4"
MODEL_VERSION = "v1.0"
MODEL_FULL_NAME = f"{MODEL_NAME}-{MODEL_VERSION}"

# Language to Expert Mapping
LANGUAGES = {
    0: "english",
    1: "hindi",
    2: "tamil",
    3: "telugu",
    4: "bengali",
    5: "marathi",
    6: "gujarati",
    7: "kannada"
}

# Language codes for datasets
LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "bengali": "bn",
    "marathi": "mr",
    "gujarati": "gu",
    "kannada": "kn"
}

# Model Architecture Configuration for 3B Model
# Adjusted for efficient NVFP4 4-bit training on RTX 5090
MODEL_CONFIG = {
    # Architecture (3B parameter model)
    "num_layers": 32,                    # 32 transformer layers
    "hidden_size": 2560,                 # 2560 hidden dimensions
    "num_attention_heads": 32,           # 32 attention heads
    "ffn_hidden_size": 10240,            # 4x hidden_size
    "max_position_embeddings": 16000,    # 16K context length
    "vocab_size": 128000,                # Tokenizer vocab (SentencePiece)

    # MoE Configuration (one expert per language)
    "num_moe_experts": 8,                # 8 language-specific experts
    "moe_router_topk": 2,                # Route to top-2 experts per token

    # NVFP4 Quantization (from NVIDIA paper arXiv:2509.25149)
    "fp4": True,                         # Enable FP4 training
    "fp4_recipe": "e2m1",                # E2M1 format: 1 sign, 2 exp, 1 mantissa
    "fp4_param": True,                   # Quantize parameters (not just activations)
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 2,                     # Micro-batch size per GPU
    "gradient_accumulation_steps": 16,   # 2 * 16 = 32 effective batch size
    "learning_rate": 1e-4,               # Initial learning rate
    "weight_decay": 0.1,                 # L2 regularization
    "warmup_steps": 2000,                # Linear warmup steps
    "max_steps": 100000,                 # Total training steps
    "save_interval": 1000,               # Save checkpoint every N steps
    "eval_interval": 500,                # Evaluate every N steps
    "log_interval": 10,                  # Log metrics every N steps

    # Learning Rate Schedule (WSD - Warmup-Stable-Decay)
    # As per NVFP4 paper: constant LR for 80% of training, decay over last 20%
    "lr_schedule": "wsd",
    "decay_start_fraction": 0.80,        # Start decay at 80% of training
}

# NVFP4 Mixed Precision Configuration
# Reference: Section 4.1 of "Pretraining Large Language Models with NVFP4"
NVFP4_CONFIG = {
    # Mixed Precision Strategy
    "mixed_precision": True,
    "fp4_linear_layers": True,           # Quantize linear layers to FP4
    "bf16_final_layers": True,           # Keep final ~15% of layers in BF16
    "num_bf16_final_blocks": 8,          # Number of final blocks to keep in BF16

    # Random Hadamard Transforms (Section 4.2)
    # Used to redistribute outliers into Gaussian distribution
    "use_hadamard_transforms": True,
    "hadamard_matrix_size": 16,          # Paper recommends 16x16
    "hadamard_apply_to": ["wgrad"],      # Only apply to weight gradients
    "hadamard_random_signs": True,       # Use random sign vectors

    # 2D Block Scaling (Section 4.3)
    # Ensures consistent quantization in forward and backward passes
    "use_2d_block_scaling": True,
    "block_scaling_size": 16,            # 16x16 blocks for weights
    "block_scaling_for_weights": True,   # 2D scaling for weights
    "block_scaling_for_activations": False,  # 1D scaling for activations

    # Stochastic Rounding (Section 4.4)
    # Reduces quantization bias in gradients
    "use_stochastic_rounding": True,
    "stochastic_rounding_on_gradients": True,
    "stochastic_rounding_on_activations": False,
    "stochastic_rounding_on_weights": False,

    # FP4 Format Parameters
    "fp4_format": "e2m1",                # E2M1: 1 sign, 2 exponent, 1 mantissa
    "fp4_block_size": 16,                # Block size for microscaling

    # Global tensor-level scaling (FP32)
    "use_tensor_scaling": True,
    "tensor_scale_format": "fp32",       # Global scale in FP32

    # Local block-level scaling (E4M3)
    "use_block_scale": True,
    "block_scale_format": "e4m3",        # Block scale in E4M3 format
}

# Data Configuration
DATA_CONFIG = {
    "max_seq_length": 16000,  # Maximum context length
    "chunk_size": 2048,        # Size for processing chunks during preprocessing
    "train_val_split": 0.995,
    "seed": 42,
    
    # Data paths
    "raw_data_dir": "/workspace/data/raw",
    "processed_data_dir": "/workspace/data/processed",
    "cache_dir": "/workspace/data/cache",
    
    # Language sampling weights (for balanced training)
    "language_weights": {
        "english": 0.30,    # 30% English
        "hindi": 0.15,      # 15% Hindi
        "tamil": 0.10,      # 10% Tamil
        "telugu": 0.10,     # 10% Telugu
        "bengali": 0.10,    # 10% Bengali
        "marathi": 0.10,    # 10% Marathi
        "gujarati": 0.075,  # 7.5% Gujarati
        "kannada": 0.075,   # 7.5% Kannada
    },
    
    # Processing configuration
    "overlap_tokens": 256,  # Overlap between chunks during preprocessing
    
    # Dataset configurations
    "datasets": {
        "phase1_pretraining": {
            # IndicCorpV2 - Large-scale Indian language corpus
            # Uses data_dir parameter with format: data/{language}_{script}
            "indiccorp_v2": {
                "name": "ai4bharat/IndicCorpV2",
                "subsets": ["hi", "ta", "te", "bn", "mr", "gu", "kn"],
                "streaming": True,
            },
            "wikipedia": {
                "name": "wikimedia/wikipedia",
                "subsets": ["20231101.en", "20231101.hi", "20231101.ta", 
                           "20231101.te", "20231101.bn", "20231101.mr", 
                           "20231101.gu", "20231101.kn"],
                "streaming": True,
            },
            "mc4": {
                "name": "allenai/c4",
                "subsets": ["en", "hi", "ta", "te", "bn", "mr", "gu", "kn"],
                "streaming": True,
            },
            "the_stack_python": {
                "name": "bigcode/the-stack-v2",
                "subset": "Python",  # Language filter, not path
                "streaming": True,
                "note": "Contains file IDs only. Actual content requires AWS credentials for Software Heritage S3"
            },
            "fineweb_edu": {
                "name": "karpathy/fineweb-edu-100b-shuffle",
                "streaming": True,
                "note": "Simple dataset with single 'text' field, no subsets needed"
            }
        },
        "phase2_instruction": {
            "aya_dataset": {
                "name": "CohereForAI/aya_dataset",
                "streaming": False,
            },
            # IndicInstruct - load multiple configs for comprehensive coverage
            "indic_instruct_dolly": {
                "name": "ai4bharat/indic-instruct-data-v0.1",
                "subset": "dolly",
                "streaming": False,
            },
            "indic_instruct_flan": {
                "name": "ai4bharat/indic-instruct-data-v0.1",
                "subset": "flan_v2",
                "streaming": False,
            },
            "indic_instruct_anudesh": {
                "name": "ai4bharat/indic-instruct-data-v0.1",
                "subset": "anudesh",
                "streaming": False,
            },
        },
        "phase3_function_calling": {
            # Only using Glaive v2 - most reliable function calling dataset
            # Berkeley and APIBench have data format issues
            "glaive_function_calling": {
                "name": "glaiveai/glaive-function-calling-v2",
                "streaming": False,
            },
        }
    }
}

# Tokenizer Configuration
TOKENIZER_CONFIG = {
    "type": "sentencepiece",
    "vocab_size": 128000,
    "model_type": "bpe",
    "character_coverage": 0.9995,
    "normalization_rule_name": "nmt_nfkc_cf",
    "split_by_unicode_script": True,
    "split_by_whitespace": True,
    "split_by_number": True,
    "max_sentence_length": 16384,
    "byte_fallback": True,
    "unk_surface": " ⁇ ",
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "min_text_length": 50,  # Minimum characters
    "max_text_length": 100000,  # Maximum characters per document
    "remove_duplicates": True,
    "deduplication_threshold": 0.85,  # Jaccard similarity threshold
    
    # Text cleaning
    "remove_urls": True,
    "remove_emails": True,
    "normalize_whitespace": True,
    "remove_control_chars": True,
    
    # Language detection
    "use_language_detection": True,
    "language_detection_threshold": 0.8,
}

# Curriculum Learning Configuration
CURRICULUM_CONFIG = {
    "enabled": True,
    "stages": [
        {
            "name": "easy",
            "steps": 10000,
            "max_seq_length": 2048,
            "description": "Short sequences, high-quality data"
        },
        {
            "name": "medium",
            "steps": 30000,
            "max_seq_length": 8000,
            "description": "Medium sequences, mixed quality"
        },
        {
            "name": "hard",
            "steps": 60000,
            "max_seq_length": 16000,
            "description": "Full sequences, all data"
        }
    ]
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_dir": "/workspace/logs",
    "tensorboard_dir": "/workspace/runs",
    "wandb_project": "indicmoe-3b-fp4",
    "wandb_entity": None,  # Set to your wandb username
    "log_level": "INFO",
}