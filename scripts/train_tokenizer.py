from pathlib import Path
import argparse

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import Lowercase


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Input text file(s).")
    parser.add_argument("--output", default="custom_tokenizer.json", help="Output tokenizer path.")
    parser.add_argument("--vocab-size", type=int, default=6000, help="Target vocabulary size.")
    parser.add_argument("--min-frequency", type=int, default=20, help="Minimum token frequency.")
    return parser.parse_args()


def main():
    args = parse_args()

    files = [str(Path(f)) for f in args.input]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Domain-specific markup tokens reserved during tokenizer training.
    special_tokens = [
        "<unk>", "<pad>", "<eos>",
        "<seq>", "</seq>", "<source>", "</source>", "<gene>", "</gene>",
        "<mRNA>", "</mRNA>", "<CDS>", "</CDS>", "<lncRNA>", "</lncRNA>",
        "<miRNA>", "</miRNA>", "<tRNA>", "</tRNA>", "<rRNA>", "</rRNA>",
        "<snRNA>", "</snRNA>", "<piRNA>", "</piRNA>", "<guide_RNA>", "</guide_RNA>",
        "<scaRNA>", "</scaRNA>", "<snoRNA>", "</snoRNA>", "<scRNA>", "</scRNA>",
        "<SRP_RNA>", "</SRP_RNA>", "<Y_RNA>", "</Y_RNA>",
        "<telomerase_RNA>", "</telomerase_RNA>", "<vault_RNA>", "</vault_RNA>",
        "<misc_RNA>", "</misc_RNA>", "<pseudo>", "</pseudo>",
        "<precursor_RNA>", "</precursor_RNA>", "<RNase_P_RNA>", "</RNase_P_RNA>",
        "<RNase_MRP_RNA>", "</RNase_MRP_RNA>", "<antisense_RNA>", "</antisense_RNA>",
        "<prim_transcript>", "</prim_transcript>", "<regulator>", "</regulator>",
        "<sig_peptide>", "</sig_peptide>",
        "<polyA_signal_sequence>", "</polyA_signal_sequence>",
        "<uORF>", "</uORF>", "<C_region>", "</C_region>", "<D_segment>", "</D_segment>",
        "<V_segment>", "</V_segment>",
        "<recoding_stimulatory_region>", "</recoding_stimulatory_region>",
        "<misc_feature>", "</misc_feature>", "<other>", "</other>",
        "<assembly_gap>", "</assembly_gap>", "<gap>", "</gap>",
        "<exon>", "</exon>", "<intron>", "</intron>",
        "<3'UTR>", "</3'UTR>", "<5'UTR>", "</5'UTR>",
        "<repeat_region>", "</repeat_region>", "<enhancer>", "</enhancer>",
        "<silencer>", "</silencer>", "<promoter>", "</promoter>",
        "<terminator>", "</terminator>",
    ]
    special_tokens = [token.lower() for token in special_tokens]

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.train(files, trainer)
    tokenizer.save(args.output)


if __name__ == "__main__":
    main()