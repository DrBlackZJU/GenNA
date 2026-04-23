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

SPECIES = "Drosophila melanogaster"
SEQUENCE_TYPE = "RNA"

NUM_SAMPLES = 2000
MAX_LEN = 4096

GENERATION_SETTINGS = [
    {"temperature": 0.55, "top_p": 0.70},
    {"temperature": 0.85, "top_p": 0.90},
    {"temperature": 1.00, "top_p": 0.95},
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


def build_prompt(sequence_type: str, species: str) -> str:
    return f"{sequence_type}, {species}, "


def sanitize_name(text: str) -> str:
    return text.lower().replace(" ", "_")


def build_output_path(
    output_dir: Path,
    sequence_type: str,
    species: str,
    temperature: float,
    top_p: float,
) -> Path:
    seq_type_str = sanitize_name(sequence_type)
    species_str = sanitize_name(species)
    temp_str = str(temperature).replace(".", "p")
    top_p_str = str(top_p).replace(".", "p")
    return output_dir / f"{seq_type_str}_{species_str}_temp_{temp_str}_top_p_{top_p_str}.txt"


@torch.inference_mode()
def generate_response(
    model,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    temperature: float,
    top_p: float,
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
        top_p=top_p,
        temperature=temperature,
        top_k=0,
        repetition_penalty=1.3,
        no_repeat_ngram_size=5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=False)


# =========================
# Main workflow
# =========================

def main():
    tokenizer = build_tokenizer(TOKENIZER_PATH)
    model = load_model(MODEL_DIR, device=device, dtype=dtype)

    prompt = build_prompt(SEQUENCE_TYPE, SPECIES)

    for setting in GENERATION_SETTINGS:
        temperature = setting["temperature"]
        top_p = setting["top_p"]
        output_path = build_output_path(
            output_dir=OUTPUT_DIR,
            sequence_type=SEQUENCE_TYPE,
            species=SPECIES,
            temperature=temperature,
            top_p=top_p,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            for _ in tqdm(
                range(NUM_SAMPLES),
                desc=f"Generating {SEQUENCE_TYPE} for {SPECIES} | temp={temperature}, top_p={top_p}",
            ):
                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                )
                cleaned = clean_text(response)
                f.write(cleaned + "\n")
                f.flush()


if __name__ == "__main__":
    main()