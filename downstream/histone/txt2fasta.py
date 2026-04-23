from __future__ import annotations

import re
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"

if not OUTPUT_DIR.exists():
    raise FileNotFoundError(f"Required path not found: {OUTPUT_DIR}")


# =========================
# Patterns
# =========================

HISTONE_FILE_PATTERN = re.compile(
    r"^histone_temp_(?P<temp>[^_]+)_top_p_(?P<top_p>[^.]+)\.txt$"
)

CDS_PATTERN = re.compile(r"<cds>(.*?)</cds>", re.IGNORECASE)
HEADER_PATTERN = re.compile(r"^(.*?)<seq>", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")


# =========================
# Utilities
# =========================

def discover_histone_files(output_dir: Path) -> list[Path]:
    matched_files = []
    for path in sorted(output_dir.glob("histone_temp_*_top_p_*.txt")):
        if HISTONE_FILE_PATTERN.match(path.name):
            matched_files.append(path)
    return matched_files


def extract_name_tokens(filename: str) -> tuple[str, str]:
    match = HISTONE_FILE_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    return match.group("temp"), match.group("top_p")


def build_output_paths(output_dir: Path, input_path: Path) -> tuple[Path, Path]:
    temp_str, top_p_str = extract_name_tokens(input_path.name)
    cds_path = output_dir / f"cds_temp_{temp_str}_top_p_{top_p_str}.fasta"
    protein_path = output_dir / f"protein_temp_{temp_str}_top_p_{top_p_str}.fasta"
    return cds_path, protein_path


def sanitize_seq_id(raw_id: str, counter: int) -> str:
    seq_id = raw_id.replace(", ", "_").replace(" ", "_").replace(",", "_")
    seq_id = re.sub(r"_+", "_", seq_id).strip("_")
    if not seq_id:
        seq_id = "Sequence"
    return f"{seq_id}_v{counter}"


def clean_cds_text(raw_cds_content: str) -> str:
    dna_seq_str = TAG_PATTERN.sub("", raw_cds_content)
    dna_seq_str = re.sub(r"\s+", "", dna_seq_str).upper()
    return dna_seq_str


def parse_generated_line(line: str, counter: int) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None

    match_cds = CDS_PATTERN.search(line)
    if match_cds is None:
        return None

    raw_cds_content = match_cds.group(1)
    dna_seq_str = clean_cds_text(raw_cds_content)

    match_header = HEADER_PATTERN.search(line)
    if match_header:
        raw_id = match_header.group(1).strip()
    else:
        raw_id = "Sequence"

    seq_id = sanitize_seq_id(raw_id, counter)
    return seq_id, dna_seq_str


def process_histone_file(input_path: Path, output_dir: Path) -> None:
    output_cds_path, output_protein_path = build_output_paths(output_dir, input_path)

    cds_records: list[SeqRecord] = []
    protein_records: list[SeqRecord] = []
    counter = 1

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_generated_line(line, counter)
            if parsed is None:
                continue

            seq_id, dna_seq_str = parsed

            try:
                cds_record = SeqRecord(
                    Seq(dna_seq_str),
                    id=seq_id,
                    description="extracted CDS nucleotide sequence",
                )
                cds_records.append(cds_record)

                protein_seq = Seq(dna_seq_str).translate(to_stop=True)
                protein_record = SeqRecord(
                    protein_seq,
                    id=seq_id,
                    description=f"translated from extracted CDS (length {len(dna_seq_str)} bp)",
                )
                protein_records.append(protein_record)

                counter += 1

            except Exception as exc:
                print(f"Warning: Error processing sequence {seq_id}: {exc}")

    if protein_records:
        protein_count = SeqIO.write(protein_records, output_protein_path, "fasta")
        cds_count = SeqIO.write(cds_records, output_cds_path, "fasta")

        print(f"Done: {input_path.name}")
        print(f"  Extracted sequences: {protein_count}")
        print(f"  Protein FASTA: {output_protein_path}")
        print(f"  CDS FASTA: {output_cds_path}")
    else:
        print(f"No matching CDS sequences found in: {input_path.name}")


# =========================
# Main
# =========================

def main() -> None:
    histone_files = discover_histone_files(OUTPUT_DIR)

    if not histone_files:
        print(f"No histone generation files found in: {OUTPUT_DIR}")
        return

    print(f"Found {len(histone_files)} histone generation file(s).")

    for input_path in histone_files:
        process_histone_file(input_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()