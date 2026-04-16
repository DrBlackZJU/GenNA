#!/usr/bin/env python3
from Bio import SeqIO
import sys
import gzip
from typing import List, Optional
from Bio.SeqFeature import SeqFeature

def annotate_rna_feature(record, gene_feature: SeqFeature, rna_features: List[SeqFeature]) -> str:
    """
    Annotate RNA features with appropriate XML-style tags including gene tag
    
    Args:
        record: The GenBank record
        gene_feature: The gene feature containing the RNA
        rna_features: List of RNA features to annotate
    
    Returns:
        Annotated RNA sequence with XML tags
    """
    # Get the full gene sequence
    gene_seq = str(record.seq[gene_feature.location.start:gene_feature.location.end])
    gene_seq = list(gene_seq)  # Convert to list for easy insertion
    
    # Calculate positions relative to gene start
    gene_start = gene_feature.location.start
    
    # Prepare annotations for feature parts
    annotations = []
    
    # Add gene tags (outermost) with a special feature index of -1
    annotations.append((0, "<gene>", "open", -1))
    annotations.append((len(gene_seq), "</gene>", "close", -1))
    
    # Process each RNA feature with its index
    for i, feature in enumerate(rna_features):
        # Get all parts of the feature
        parts = feature.location.parts if hasattr(feature.location, "parts") else [feature.location]
        
        # For each segment in the feature
        for part in parts:
            start = part.start - gene_start
            end = part.end - gene_start
            
            # Determine the appropriate tag
            if feature.type == "CDS":
                tag = "CDS"
            elif feature.type == "ncRNA":
                tag = feature.qualifiers.get("ncRNA_class", ["ncRNA"])[0]
            elif feature.type == "regulatory":
                # Use regulatory_class if available, otherwise just "regulatory"
                tag = feature.qualifiers.get("regulatory_class", ["regulatory"])[0]
            elif feature.type == "sig_peptide":
                tag = "sig_peptide"
            else:
                tag = feature.type
            
            # Add opening and closing tags with the feature's index
            annotations.append((start, f"<{tag}>", "open", i))
            annotations.append((end, f"</{tag}>", "close", i))
    
    # Sort annotations by:
    # 1. Position
    # 2. Opening tags before closing tags at the same position
    # 3. For opening tags: order by feature index (earlier features first)
    # 4. For closing tags: order by reverse feature index (later features first)
    annotations.sort(key=lambda x: (
        x[0], 
        0 if x[2] == "open" else 1, 
        x[3] if x[2] == "open" else -x[3]
    ))
    
    # Insert tags from end to beginning to maintain correct positions
    for item in reversed(annotations):
        pos, tag, _, _ = item
        gene_seq.insert(pos, tag)
    
    return "".join(gene_seq)

def get_description(record, gene_feature: SeqFeature, rna_features: List[SeqFeature]) -> str:
    """
    Generate descriptive header line with organism and gene/RNA info
    
    Args:
        record: The GenBank record
        gene_feature: The gene feature
        rna_features: List of RNA features
    
    Returns:
        Description string
    """
    species = record.annotations.get("organism", "unknown species")
    gene_name = gene_feature.qualifiers.get("gene", ["unknown gene"])[0]
    
    # Get product description from first RNA feature
    product = "unknown product"
    if rna_features:
        first_rna = rna_features[0]
        if "product" in first_rna.qualifiers:
            product = first_rna.qualifiers["product"][0]
        elif "note" in first_rna.qualifiers:
            product = first_rna.qualifiers["note"][0]
    
    return f"RNA, {species}, {gene_name}, {product}"

def process_rna_genbank_record(record, out_handle):
    """
    Process a single RNA GenBank record and write output
    
    Args:
        record: The GenBank record to process
        out_handle: Output file handle
    """
    # First get all gene features
    gene_features = [f for f in record.features if f.type == "gene"]
    
    for gene_feature in gene_features:
        # Find all features within this gene that we want to annotate
        features_to_annotate = []
        for feature in record.features:
            # Check if feature is within this gene and is one of the types we want to annotate
            if (feature.location.start >= gene_feature.location.start and
                feature.location.end <= gene_feature.location.end and
                feature.type in ["CDS", "ncRNA", "tRNA", "rRNA", "precursor_RNA", "sig_peptide", "regulatory"]):
                features_to_annotate.append(feature)
        
        if not features_to_annotate:
            continue
        
        # Annotate the gene with its features
        annotated_seq = annotate_rna_feature(record, gene_feature, features_to_annotate)
        
        # Generate and write description and sequence
        description = get_description(record, gene_feature, features_to_annotate)
        out_handle.write(f"{description}<seq>{annotated_seq}</seq><eos>\n")

def main():
    """Main function that handles command line arguments and file processing"""
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        print("Input file can be gzipped (.gz extension)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Handle gzipped input
    if input_file.endswith('.gz'):
        try:
            with gzip.open(input_file, 'rt') as gz_handle:
                with open(output_file, "w") as out_handle:
                    for record in SeqIO.parse(gz_handle, "genbank"):
                        process_rna_genbank_record(record, out_handle)
        except Exception as e:
            print(f"Error processing gzipped file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            with open(input_file) as in_handle:
                with open(output_file, "w") as out_handle:
                    for record in SeqIO.parse(in_handle, "genbank"):
                        process_rna_genbank_record(record, out_handle)
        except Exception as e:
            print(f"Error processing file: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()