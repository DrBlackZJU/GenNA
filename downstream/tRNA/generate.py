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
OUTPUT_PATH = PROJECT_ROOT / "output" / "tRNA.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

flash_attn_available = (
    device.type == "cuda" and importlib.util.find_spec("flash_attn") is not None
)

MAX_LEN = 4096
NUM_SAMPLES_PER_PROMPT = 500

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

TRNA_ITEMS = [
    # Alanine
    "trnaA-AGC, trna-Ala",
    "trnaA-GGC, trna-Ala",
    "trnaA-UGC, trna-Ala",
    "trnaA-CGC, trna-Ala",

    # Cysteine
    "trnaC-ACA, trna-Cys",
    "trnaC-GCA, trna-Cys",

    # Aspartic Acid
    "trnaD-AUC, trna-Asp",
    "trnaD-GUC, trna-Asp",

    # Glutamic Acid
    "trnaE-UUC, trna-Glu",
    "trnaE-CUC, trna-Glu",

    # Phenylalanine
    "trnaF-AAA, trna-Phe",
    "trnaF-GAA, trna-Phe",

    # Glycine
    "trnaG-ACC, trna-Gly",
    "trnaG-GCC, trna-Gly",
    "trnaG-UCC, trna-Gly",
    "trnaG-CCC, trna-Gly",

    # Histidine
    "trnaH-AUG, trna-His",
    "trnaH-GUG, trna-His",

    # Isoleucine
    "trnaI-AAU, trna-Ile",
    "trnaI-GAU, trna-Ile",
    "trnaI-UAU, trna-Ile",

    # Lysine
    "trnaK-UUU, trna-Lys",
    "trnaK-CUU, trna-Lys",

    # Leucine
    "trnaL-UAA, trna-Leu",
    "trnaL-CAA, trna-Leu",
    "trnaL-AAG, trna-Leu",
    "trnaL-GAG, trna-Leu",
    "trnaL-UAG, trna-Leu",
    "trnaL-CAG, trna-Leu",

    # Methionine
    "trnaM-CAU, trna-Met",

    # Asparagine
    "trnaN-AUU, trna-Asn",
    "trnaN-GUU, trna-Asn",

    # Proline
    "trnaP-AGG, trna-Pro",
    "trnaP-GGG, trna-Pro",
    "trnaP-UGG, trna-Pro",
    "trnaP-CGG, trna-Pro",

    # Glutamine
    "trnaQ-UUG, trna-Gln",
    "trnaQ-CUG, trna-Gln",

    # Arginine
    "trnaR-ACG, trna-Arg",
    "trnaR-GCG, trna-Arg",
    "trnaR-UCG, trna-Arg",
    "trnaR-CCG, trna-Arg",
    "trnaR-UCU, trna-Arg",
    "trnaR-CCU, trna-Arg",

    # Serine
    "trnaS-AGA, trna-Ser",
    "trnaS-GGA, trna-Ser",
    "trnaS-UGA, trna-Ser",
    "trnaS-CGA, trna-Ser",
    "trnaS-ACU, trna-Ser",
    "trnaS-GCU, trna-Ser",

    # Threonine
    "trnaT-AGU, trna-Thr",
    "trnaT-GGU, trna-Thr",
    "trnaT-UGU, trna-Thr",
    "trnaT-CGU, trna-Thr",

    # Valine
    "trnaV-AAC, trna-Val",
    "trnaV-GAC, trna-Val",
    "trnaV-UAC, trna-Val",
    "trnaV-CAC, trna-Val",

    # Tryptophan
    "trnaW-CCA, trna-Trp",

    # Tyrosine
    "trnaY-AUA, trna-Tyr",
    "trnaY-GUA, trna-Tyr",
]


def build_prompt(item: str) -> str:
    return f"Genomic DNA, {item}<seq>"


# =========================
# Main workflow
# =========================

def main():
    tokenizer = build_tokenizer(TOKENIZER_PATH)
    model = load_model(MODEL_DIR, device=device, dtype=dtype)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in TRNA_ITEMS:
            prompt = build_prompt(item)

            for _ in tqdm(range(NUM_SAMPLES_PER_PROMPT), desc=f"Generating for {item}"):
                response = generate_response(model, tokenizer, prompt)
                cleaned = clean_text(response)
                f.write(cleaned + "\n")
                f.flush()


if __name__ == "__main__":
    main()