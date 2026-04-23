from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
)

# =========================
# Paths and runtime setup
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model" / "GenNA"
TOKENIZER_PATH = PROJECT_ROOT / "configs" / "tokenizer.json"
OUTPUT_PATH = PROJECT_ROOT / "output" / "rRNA.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

flash_attn_available = (
    device.type == "cuda" and importlib.util.find_spec("flash_attn") is not None
)

MAX_LEN = 4096
NUM_SAMPLES_PER_PROMPT = 1000

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Required path not found: {MODEL_DIR}")

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Required path not found: {TOKENIZER_PATH}")


# =========================
# Tokenizer
# =========================

def build_tokenizer(tokenizer_path: Path) -> PreTrainedTokenizerFast:
    tokenizer_core = Tokenizer.from_file(str(tokenizer_path))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    return tokenizer


# =========================
# Model
# =========================

def load_model(model_dir: Path, device: torch.device, dtype: torch.dtype):
    config = AutoConfig.from_pretrained(model_dir)

    model_kwargs = {
        "config": config,
        "torch_dtype": dtype,
    }

    if flash_attn_available:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs).to(device)
    model.eval()
    return model


# =========================
# Generation
# =========================

def clean_text(text: str) -> str:
    return text.replace(" ", "").replace("Ġ", " ")


@torch.inference_mode()
def generate_response(
    model,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_length: int = MAX_LEN,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        top_k=0,
        repetition_penalty=1.3,
        no_repeat_ngram_size=5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=False)


# =========================
# Input prompts
# =========================

RRNA_ITEMS = [
    "5S ribosomal RNA",
    "5.8S ribosomal RNA",
    "18S ribosomal RNA",
    "28S ribosomal RNA",
]


def build_prompt(item: str) -> str:
    return f"RNA, {item}<seq>"


# =========================
# Main workflow
# =========================

def main():
    tokenizer = build_tokenizer(TOKENIZER_PATH)
    model = load_model(MODEL_DIR, device=device, dtype=dtype)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in RRNA_ITEMS:
            prompt = build_prompt(item)

            for _ in tqdm(range(NUM_SAMPLES_PER_PROMPT), desc=f"Generating for {item}"):
                response = generate_response(model, tokenizer, prompt)
                cleaned = clean_text(response)
                f.write(cleaned + "\n")
                f.flush()


if __name__ == "__main__":
    main()