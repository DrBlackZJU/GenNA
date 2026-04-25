<!-- fullWidth: false tocVisible: false tableWrap: true -->
# GenNA: Conditional Generation of Nucleotide Sequences Guided by Natural-Language Annotations

## Overview

**GenNA** is a generative nucleotide foundation model for conditional DNA/RNA sequence generation guided by natural-language annotations. The model is trained in an autoregressive manner on a multimodal corpus that combines nucleotide sequences, natural-language functional descriptions, species information, gene-related metadata, and XML-style structural annotations.

Compared with conventional nucleotide language models that mainly rely on sequence context or structured labels, GenNA is designed to directly use natural language as a conditioning signal, enabling more flexible and human-readable sequence design.

![GenNA workflow](images/workflow.png "GenNA workflow")

## Environment Setup

We recommend creating a clean conda environment before running the project:

```bash
conda create -n genna python=3.10
conda activate genna
pip install -r requirements.txt
```

## Pretraining Data

GenNA is designed to be pretrained from full-format **GenBank** files. The pretraining data in the manuscript were derived from the **NCBI RefSeq** database.

RefSeq root directory:  `https://ftp.ncbi.nlm.nih.gov/genomes/refseq/`

For example, for **Homo sapiens**, you may download:

- **Genomic DNA**: `https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/reference/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.gbff.gz`
- **RNA**: `https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/reference/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_rna.gbff.gz`

You may also use your own GenBank files to build a custom multimodal dataset.

## Data Preprocessing

To convert GenBank files into the text format used for model training, run:

For **Genomic DNA**:

```bash
python scripts/genome_preprocess.py input.gbff.gz output.txt
```

For **RNA**:

```bash
python scripts/rna_preprocess.py input.gbff.gz output.txt
```

An example for what our pretraining corpus looks like is available at `data/sample.txt`.

## Model Weights

Available pretrained checkpoints:

- **GenNA-3.6B**: `https://huggingface.co/DrBlack/GenNA`
- **GenNA-0.36B**: `https://huggingface.co/DrBlack/GenNA-small`

After downloading, place the checkpoint directories under `model/`.

## Pretraining

To reproduce the pretraining pipeline or train on your own dataset:

```bash
python train.py configs/train_GenNA.json
```

or

```bash
python train.py configs/train_GenNA_small.json
```

Before training, replace the dataset path in the config file `"train_file": "data/sample.txt"` with your actual training file path.

## Quick Inference Test

To quickly verify that the model can load and generate sequences correctly, open the notebook `test/test_generation.ipynb`

If `streamlit` is installed, you can also launch the web demo:

```bash
streamlit run test/web.py
```

![Web demo](images/web.png "Web demo")

## Interpretability Analysis

To better understand what GenNA has learned beyond generation, we provide several interpretability analyses corresponding to the experiments described in our manuscript.

Run the `.ipynb` scripts under `interpret/` to reproduce those figures  in our manuscript. 

## Generation Tasks

In our manuscript, we completed several generation experiments including:

- unconditional self-guided generation
- species-specific generation
- targeted generation of structured non-coding RNAs such as **tRNAs** and **rRNAs**
- targeted generation of protein-coding sequences such as **histones**
- downstream evaluation and visualization of generated outputs

Most of these scripts are available under `downstream/`.

## Example: tRNA Generation

To run the tRNA generation experiment:

```bash
python downstream/tRNA/generate.py
```

This produces the output file `outputs/tRNA.txt`.

Then convert the generated text file to FASTA format:

```bash
python downstream/tRNA/txt2fasta.py
```

Install **tRNAscan-SE** and evaluate the generated sequences:

```bash
conda install -c bioconda trnascan-se
tRNAscan-SE -E outputs/tRNA.fasta -o outputs/tRNAscan.txt
```

After that, fill the output path into `downstream/tRNA/visualize.ipynb` and run the notebook to obtain the visualization results.

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article {Shen2026.04.22.720063,
	author = {Shen, Yi and Cao, Guangshuo and Wu, Jianghong and Chen, Dijun and Feng, Cong and Chen, Ming},
	title = {GenNA: Conditional generation of nucleotide sequences guided by natural-language annotations},
	elocation-id = {2026.04.22.720063},
	year = {2026},
	doi = {10.64898/2026.04.22.720063},
	URL = {https://www.biorxiv.org/content/early/2026/04/24/2026.04.22.720063},
	eprint = {https://www.biorxiv.org/content/early/2026/04/24/2026.04.22.720063.full.pdf},
	journal = {bioRxiv}
}```
