import torch
import warnings
from utils.utils import *
from config import *
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

warnings.filterwarnings(action='ignore')

def load_model(size):

    """
    model selection
    """
    
    # Phantom Bit
    bit_quant_skip = ["linear_q", "linear_k", "linear_v", "linear_o", "gating_phantom_1", "gating_phantom_2"]
    # Vision target modules
    if size == '7b':
        from .arch_7b.modeling_phantom import PhantomForCausalLM
        from .arch_7b.tokenization_internlm2 import InternLM2Tokenizer as PhantomTokenizer
        path = MODEL_7B
        bit_quant_skip += ["mlp1", "wqkv", "output"]

        # Loading tokenizer
        tokenizer = PhantomTokenizer.from_pretrained(path, padding_side='left')

        # bits
        bits = 8

    elif size == '3.8b':
        from .arch_3_8b.modeling_phantom import PhantomForCausalLM
        path = MODEL_3_8B
        bit_quant_skip += ["mlp1", "qkv_proj", "phantom", "lm_head"]

        # Loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')

        # bits
        bits = 8

    elif size == '1.8b':
        from .arch_1_8b.modeling_phantom import PhantomForCausalLM 
        from .arch_1_8b.tokenization_internlm2 import InternLM2Tokenizer as PhantomTokenizer
        path = MODEL_1_8B
        bit_quant_skip += ["mlp1", "wqkv", "phantom", "output"]

        # Loading tokenizer
        tokenizer = PhantomTokenizer.from_pretrained(path, padding_side='left')

        # bits
        bits = 8

    elif size == '0.5b':
        from .arch_0_5b.modeling_phantom import PhantomForCausalLM 
        path = MODEL_0_5B
        bit_quant_skip += ["mlp1", "q_proj", "k_proj", "v_proj", "phantom", "lm_head"]

        # Loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')

        # bits
        bits = 8
    else:
        raise Exception("Unsupported Size")


    # huggingface model configuration
    huggingface_config = {}

    # Bit quantization
    if bits in [4, 8]:
        huggingface_config.update(dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_skip_modules=bit_quant_skip,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        ))
    else:
        huggingface_config.update(dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ))

    # Model Uploading
    model = PhantomForCausalLM.from_pretrained(path, **huggingface_config)

    # Parameter arrangement
    freeze_model(model)
    model.eval()

    # bfloat16/float16 conversion 
    for param in model.parameters():
        if 'float32' in str(param.dtype).lower() or 'float16' in str(param.dtype).lower():
            param.data = param.data.to(torch.bfloat16)
    
    return model, tokenizer