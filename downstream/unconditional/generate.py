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
OUTPUT_DIR = PROJECT_ROOT / "output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

flash_attn_available = (
    device.type == "cuda" and importlib.util.find_spec("flash_attn") is not None
)

MAX_LEN = 4096
NUM_SAMPLES_PER_TASK = 10000

GENERATION_TASKS = [
    {
        "name": "RNA",
        "prompt": "RNA, ",
        "output_path": OUTPUT_DIR / "rna_unconditional.txt",
    },
    {
        "name": "Genomic DNA",
        "prompt": "Genomic DNA, ",
        "output_path": OUTPUT_DIR / "dna_unconditional.txt",
    },
]

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Required path not found: {MODEL_DIR}")

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Required path not found: {TOKENIZER_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def run_generation_task(
    model,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    output_path: Path,
    num_samples: int,
    task_name: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in tqdm(range(num_samples), desc=f"Generating {task_name}"):
            response = generate_response(model, tokenizer, prompt)
            cleaned = clean_text(response)
            f.write(cleaned + "\n")
            f.flush()


# =========================
# Main workflow
# =========================

def main():
    tokenizer = build_tokenizer(TOKENIZER_PATH)
    model = load_model(MODEL_DIR, device=device, dtype=dtype)

    for task in GENERATION_TASKS:
        run_generation_task(
            model=model,
            tokenizer=tokenizer,
            prompt=task["prompt"],
            output_path=task["output_path"],
            num_samples=NUM_SAMPLES_PER_TASK,
            task_name=task["name"],
        )


if __name__ == "__main__":
    main()