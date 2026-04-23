from __future__ import annotations

import re
from pathlib import Path


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "outputs" / "tRNA.txt"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tRNA.fasta"

LINE_WIDTH = 60

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Required path not found: {INPUT_PATH}")


# =========================
# Patterns
# =========================

DESC_PATTERN = re.compile(r"^genomic dna,\s*(.*?)<seq>", re.IGNORECASE)
SEQ_PATTERN = re.compile(r"<seq>(.*?)</seq>", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")
NON_BASE_PATTERN = re.compile(r"[^ACGTU]")
WHITESPACE_PATTERN = re.compile(r"\s+")


# =========================
# Utilities
# =========================

def wrap_seq(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def clean_sequence(seq_block: str) -> str:
    seq = TAG_PATTERN.sub("", seq_block)
    seq = WHITESPACE_PATTERN.sub("", seq)
    seq = seq.upper()
    seq = NON_BASE_PATTERN.sub("", seq)
    return seq


def sanitize_description(desc: str, lineno: int) -> str:
    desc = desc.strip()
    if not desc:
        desc = f"seq_{lineno}"
    desc = desc.replace(" ", "")
    return desc


def parse_line(line: str, lineno: int) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None

    match_desc = DESC_PATTERN.search(line)
    if match_desc is None:
        print(f"[Warning] Line {lineno}: missing description block, skipped.")
        return None

    desc = sanitize_description(match_desc.group(1), lineno)

    match_seq = SEQ_PATTERN.search(line)
    if match_seq is None:
        print(f"[Warning] Line {lino}: missing <seq> block, skipped.")
        return None

    seq_block = match_seq.group(1)
    seq = clean_sequence(seq_block)

    if not seq:
        print(f"[Warning] Line {lineno}: sequence is empty after cleaning, skipped.")
        return None

    return desc, seq


# =========================
# Main
# =========================

def main() -> None:
    record_count = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, start=1):
            parsed = parse_line(line, lineno)
            if parsed is None:
                continue

            desc, seq = parsed
            record_count += 1

            fasta_id = f"{desc}_{record_count}"
            fout.write(f">{fasta_id}\n")
            fout.write(wrap_seq(seq, LINE_WIDTH) + "\n")

    print(f"Done: converted {record_count} sequence(s).")
    print(f"Output FASTA: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()