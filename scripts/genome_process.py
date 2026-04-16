#!/usr/bin/env python3
from Bio import SeqIO
import random
import sys
import gzip
import os
import tempfile
from typing import List, Tuple, Optional
from Bio.SeqFeature import SeqFeature

def annotate_feature(record, gene_feature: SeqFeature, child_features: List[SeqFeature], is_pseudo: bool = False) -> str:
    """
    Annotate genomic features with XML-style tags, ensuring proper nesting order
    
    Args:
        record: The GenBank record
        gene_feature: The gene feature to annotate
        child_features: List of child features to annotate
        is_pseudo: Whether the gene is a pseudogene
    
    Returns:
        Annotated gene sequence with XML tags
    """
    # Get the full gene sequence
    gene_seq = str(record.seq[gene_feature.location.start:gene_feature.location.end])
    gene_seq = list(gene_seq)  # Convert to list for easy insertion
    
    # Calculate positions relative to gene start
    gene_start = gene_feature.location.start
    
    # Prepare annotations with tracking indexes
    annotations = []
    
    # Add gene tags first (special index -1)
    annotations.append((0, "<gene>", "open", -1))
    annotations.append((len(gene_seq), "</gene>", "close", -1))
    
    # Process child features with their creation order
    for feature_idx, child_feature in enumerate(child_features):
        # Get all parts of the feature
        parts = child_feature.location.parts if hasattr(child_feature.location, "parts") else [child_feature.location]
        
        for part in parts:
            start = part.start - gene_start
            end = part.end - gene_start
            
            # Determine tag type
            if is_pseudo:
                tag = "pseudo"
            else:
                tag = child_feature.type
                if tag == "ncRNA" and "ncRNA_class" in child_feature.qualifiers:
                    tag = child_feature.qualifiers["ncRNA_class"][0]

            # Record both opening and closing tags with feature index
            annotations.append((start, f"<{tag}>", "open", feature_idx))
            annotations.append((end, f"</{tag}>", "close", feature_idx))
    
    # Sorting logic:
    # 1. Primary key: position
    # 2. Open tags before close tags at same position
    # 3. For opens: earlier features first (ascending index)
    # 4. For closes: later features first (descending index)
    annotations.sort(key=lambda x: (
        x[0],  # Position
        0 if x[2] == "open" else 1,  # Open before close
        x[3] if x[2] == "open" else -x[3]  # Open order vs close order
    ))
    
    # Insert tags from end to beginning to maintain positions
    for item in reversed(annotations):
        pos, tag, _, _ = item
        gene_seq.insert(pos, tag)
    
    return "".join(gene_seq)

def get_extended_sequence(record, gene_feature: SeqFeature, left_extend: int, right_extend: int) -> Tuple[str, int, int]:
    """
    Extend gene sequence with flanking regions while respecting boundaries
    
    Args:
        record: The GenBank record
        gene_feature: The gene feature to extend
        left_extend: Number of bases to extend on the left
        right_extend: Number of bases to extend on the right
    
    Returns:
        Tuple of (extended sequence, gene start in extended, gene end in extended)
    """
    # Calculate extended coordinates (ensure within record boundaries)
    extended_start = max(0, gene_feature.location.start - left_extend)
    extended_end = min(len(record.seq), gene_feature.location.end + right_extend)
    
    # Get the extended sequence
    extended_seq = str(record.seq[extended_start:extended_end])
    
    # Calculate the original gene position within extended sequence
    gene_start_in_extended = gene_feature.location.start - extended_start
    gene_end_in_extended = gene_feature.location.end - extended_start
    
    return extended_seq, gene_start_in_extended, gene_end_in_extended

def get_description(record, gene_feature: SeqFeature, first_child_feature: Optional[SeqFeature]) -> str:
    """
    Generate descriptive header line with organism, gene and product info
    
    Args:
        record: The GenBank record
        gene_feature: The gene feature
        first_child_feature: First child feature of the gene
    
    Returns:
        Description string
    """
    species = record.annotations.get("organism", "unknown species")
    gene_name = gene_feature.qualifiers.get("gene", ["unknown gene"])[0]
    
    # Get product description from first child feature if available
    product = "unknown product"
    if first_child_feature:
        if "product" in first_child_feature.qualifiers:
            product = first_child_feature.qualifiers["product"][0]
        elif "note" in first_child_feature.qualifiers:
            product = first_child_feature.qualifiers["note"][0]
    
    return f"Genomic DNA, {species}, {gene_name}, {product}"

def is_pseudogene(gene_feature: SeqFeature) -> bool:
    """
    Determine if a gene feature is a pseudogene
    
    Args:
        gene_feature: The gene feature to check
    
    Returns:
        True if the gene is a pseudogene, False otherwise
    """
    return ("pseudo" in gene_feature.qualifiers or 
            gene_feature.type == "pseudogene" or
            ("gene_biotype" in gene_feature.qualifiers and 
             "pseudogene" in gene_feature.qualifiers["gene_biotype"][0]))

def process_genbank_record(record, out_handle):
    """
    Process a single GenBank record and write output
    
    Args:
        record: The GenBank record to process
        out_handle: Output file handle
    """
    
    # First pass: collect all gene features
    gene_features = [f for f in record.features if f.type == "gene"]
    
    for gene_feature in gene_features:
        gene_name = gene_feature.qualifiers.get("gene", ["unknown"])[0]
        
        # Check if this is a pseudogene
        is_pseudo = is_pseudogene(gene_feature)
        
        # Find all child features (any type except gene)
        child_features = []
        for feature in record.features:
            # Check if feature is within this gene and not another gene
            if (feature.location.start >= gene_feature.location.start and
                feature.location.end <= gene_feature.location.end and
                feature.type != "gene"):
                child_features.append(feature)
        
        if not child_features:
            continue
        
        first_child_feature = child_features[0]
        
        # Determine which features to annotate
        features_to_annotate = []
        if is_pseudo:
            features_to_annotate = [first_child_feature]
        else:
            if first_child_feature.type == "mRNA":
                features_to_annotate.append(first_child_feature)
                # Find first CDS feature
                for feature in child_features:
                    if feature.type == "CDS":
                        features_to_annotate.append(feature)
                        break
            elif first_child_feature.type == "precursor_RNA":
                features_to_annotate.append(first_child_feature)
                # Find the ncRNA feature that follows
                for feature in child_features[1:]:
                    if feature.type == "ncRNA":
                        features_to_annotate.append(feature)
                        break
            else:
                features_to_annotate.append(first_child_feature)
        
        if features_to_annotate:
            # Annotate the gene features (now includes gene tags)
            annotated_gene_seq = annotate_feature(record, gene_feature, features_to_annotate, is_pseudo)
            
            # Independently extend 0-100 nucleotides on each side
            left_extend = random.randint(0, 100)
            right_extend = random.randint(0, 100)
            
            # Get the extended sequence with gene tags
            extended_seq, gene_start, gene_end = get_extended_sequence(
                record, gene_feature, left_extend, right_extend
            )
            
            # Construct the final sequence with <seq> tags
            final_seq = (
                extended_seq[:gene_start] +
                annotated_gene_seq +
                extended_seq[gene_end:]
            )
            
            # Generate and write description and sequence
            description = get_description(record, gene_feature, first_child_feature)
            out_handle.write(f"{description}<seq>{final_seq}</seq><eos>\n")

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
                        process_genbank_record(record, out_handle)
        except Exception as e:
            print(f"Error processing gzipped file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            with open(input_file) as in_handle:
                with open(output_file, "w") as out_handle:
                    for record in SeqIO.parse(in_handle, "genbank"):
                        process_genbank_record(record, out_handle)
        except Exception as e:
            print(f"Error processing file: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()