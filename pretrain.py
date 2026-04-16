from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Features, Value, load_dataset
from tokenizers import Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


# =========================
# Dataclass definitions
# =========================

@dataclass
class ModelArguments:
    # Existing model or checkpoint source.
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained model/checkpoint for continuing training."},
    )

    # Configuration source for random initialization.
    config_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model config directory (containing config.json) for training from scratch."},
    )

    tokenizer_file: str = field(
        default="configs/custom_tokenizer.json",
        metadata={"help": "Path to tokenizer JSON file produced by tokenizers library."},
    )

    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Attention backend, e.g. flash_attention_2 / sdpa / eager."},
    )

    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "One of: float32, float16, bfloat16."},
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading model/config."},
    )

    def __post_init__(self):
        if self.model_name_or_path and self.config_name_or_path:
            raise ValueError("`model_name_or_path` and `config_name_or_path` are mutually exclusive.")
        if not self.model_name_or_path and not self.config_name_or_path:
            raise ValueError("One of `model_name_or_path` or `config_name_or_path` must be provided.")


@dataclass
class DataArguments:
    train_file: str = field(
        default="data/part.txt",
        metadata={"help": "Training text file path."},
    )

    sample_by: str = field(
        default="line",
        metadata={"help": "How text loader samples file. Commonly 'line'."},
    )

    streaming: bool = field(
        default=True,
        metadata={"help": "Use streaming dataset."},
    )

    max_length: int = field(
        default=4096,
        metadata={"help": "Max sequence length."},
    )

    padding_side: str = field(
        default="left",
        metadata={"help": "Tokenizer padding side: left or right."},
    )

    add_special_tokens: bool = field(
        default=True,
        metadata={"help": "Whether tokenizer adds special tokens."},
    )

    text_column_name: str = field(
        default="text",
        metadata={"help": "Column name produced by text dataset loader."},
    )


# =========================
# Utility functions
# =========================

def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[dtype_name]


def validate_paths(model_args: ModelArguments, data_args: DataArguments):
    if not Path(model_args.tokenizer_file).exists():
        raise FileNotFoundError(f"Tokenizer file not found: {model_args.tokenizer_file}")

    if not Path(data_args.train_file).exists():
        raise FileNotFoundError(f"Train file not found: {data_args.train_file}")

    if model_args.model_name_or_path and not Path(model_args.model_name_or_path).exists():
        raise FileNotFoundError(f"Model path not found: {model_args.model_name_or_path}")

    if model_args.config_name_or_path and not Path(model_args.config_name_or_path).exists():
        raise FileNotFoundError(f"Config path not found: {model_args.config_name_or_path}")


def build_tokenizer(model_args: ModelArguments, data_args: DataArguments) -> PreTrainedTokenizerFast:
    tk = Tokenizer.from_file(model_args.tokenizer_file)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )

    tokenizer.padding_side = data_args.padding_side
    tokenizer.model_max_length = data_args.max_length

    # BOS remains unset unless it is required by the tokenizer or model setup.
    # tokenizer.bos_token = "<bos>"

    return tokenizer


def build_config(model_args: ModelArguments, tokenizer: PreTrainedTokenizerFast):
    config_source = model_args.model_name_or_path or model_args.config_name_or_path

    config = AutoConfig.from_pretrained(
        config_source,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Align tokenizer-dependent fields in the model config.
    tokenizer_size = len(tokenizer)
    if getattr(config, "vocab_size", None) != tokenizer_size:
        logger.warning(
            "config.vocab_size (%s) != tokenizer size (%s). Override config.vocab_size -> %s",
            getattr(config, "vocab_size", None),
            tokenizer_size,
            tokenizer_size,
        )
        config.vocab_size = tokenizer_size

    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "bos_token_id", None) is not None:
        config.bos_token_id = tokenizer.bos_token_id

    return config


def build_model(
    model_args: ModelArguments,
    config,
):
    dtype = resolve_dtype(model_args.torch_dtype)

    if model_args.model_name_or_path:
        logger.info("Loading pretrained weights from: %s", model_args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
        )
    else:
        logger.info("Initializing model from config (random init): %s", model_args.config_name_or_path)
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
        )

    return model


def preprocess_batch(batch, tokenizer: PreTrainedTokenizerFast, data_args: DataArguments):
    texts = batch[data_args.text_column_name]

    encodings = tokenizer(
        texts,
        max_length=data_args.max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=data_args.add_special_tokens,
    )

    input_ids = encodings["input_ids"]

    labels = [
        [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq]
        for seq in input_ids
    ]
    encodings["labels"] = labels
    return encodings


def build_train_dataset(data_args: DataArguments, tokenizer: PreTrainedTokenizerFast):
    dataset = load_dataset(
        "text",
        data_files={"train": data_args.train_file},
        split="train",
        sample_by=data_args.sample_by,
        streaming=data_args.streaming,
        features=Features({data_args.text_column_name: Value("string")}),
    )

    dataset = dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer, data_args),
        batched=True,
        remove_columns=[data_args.text_column_name],
    )

    return dataset


def log_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Total parameters: %.2fM", total_params / 1e6)
    logger.info("Trainable parameters: %.2fM", trainable_params / 1e6)


def main():
    model_args, data_args, training_args = parse_args()
    setup_logging()

    validate_paths(model_args, data_args)
    set_seed(training_args.seed)

    tokenizer = build_tokenizer(model_args, data_args)
    config = build_config(model_args, tokenizer)
    train_dataset = build_train_dataset(data_args, tokenizer)
    model = build_model(model_args, config)

    # Cache is disabled when gradient checkpointing is enabled.
    if training_args.gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    log_model_stats(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_state()

    logger.info("Training finished. Metrics: %s", train_result.metrics)


if __name__ == "__main__":
    main()