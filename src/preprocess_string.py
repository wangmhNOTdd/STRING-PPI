#!/usr/bin/env python3
"""
Preprocess STRING database files:
1. Filter protein physical links by combined_score
2. Map protein IDs using aliases file
3. Align sequences with links and info
4. Remove self-loops and duplicate edges
"""

import argparse
import gzip
import os
import pandas as pd
from pathlib import Path


def read_gzipped_file(filepath, sep='\t', comment='#'):
    """Read a gzipped file into a DataFrame"""
    with gzip.open(filepath, 'rt') as f:
        df = pd.read_csv(f, sep=sep, comment=comment)
        # If there's only one column, it means the separator was not found
        # and we need to split the column by spaces
        if len(df.columns) == 1:
            # Split the single column by spaces
            df = df[df.columns[0]].str.split(' ', expand=True)
            # Set the correct column names
            df.columns = ['protein1', 'protein2', 'combined_score']
            # Convert combined_score to integer
            df['combined_score'] = df['combined_score'].astype(int)
        return df


def filter_links(links_df, min_score=700):
    """Filter links by combined_score"""
    return links_df[links_df['combined_score'] >= min_score]


def map_protein_ids(aliases_file, preferred_id_type='ensp'):
    """
    Map protein IDs from aliases file
    preferred_id_type: 'ensp', 'preferred_name', or 'uniprot'
    """
    aliases_df = read_gzipped_file(aliases_file)

    # Set the correct column names
    aliases_df.columns = ['protein_external_id', 'alias', 'source']

    # Filter for preferred ID type
    if preferred_id_type == 'ensp':
        # Map to Ensembl Protein IDs (format: 9606.ENSP...)
        # For ENSP, we can just use the protein_external_id directly since it's already in the right format
        mapped = aliases_df.drop_duplicates('protein_external_id').copy()
        mapped['preferred_id'] = mapped['protein_external_id']
    elif preferred_id_type == 'preferred_name':
        # Map to preferred names
        mapped = aliases_df.drop_duplicates('protein_external_id').copy()
        mapped['preferred_id'] = mapped['protein_external_id']  # Use protein_external_id as it contains the ENSP IDs
    elif preferred_id_type == 'uniprot':
        # Map to UniProt IDs
        mapped = aliases_df[aliases_df['source'].str.contains('UniProt', case=False, na=False)].copy()
        mapped['preferred_id'] = mapped['alias']
    else:
        raise ValueError(f"Unknown preferred_id_type: {preferred_id_type}")

    # Create mapping dictionary
    id_map = dict(zip(mapped['protein_external_id'], mapped['preferred_id']))
    return id_map


def process_links(links_file, id_map, min_score=700):
    """Process links file with ID mapping and filtering"""
    links_df = read_gzipped_file(links_file)

    # Filter by score
    links_df = filter_links(links_df, min_score)

    # Map protein IDs
    links_df['protein1_mapped'] = links_df['protein1'].map(id_map)
    links_df['protein2_mapped'] = links_df['protein2'].map(id_map)

    # Remove unmapped entries
    links_df = links_df.dropna(subset=['protein1_mapped', 'protein2_mapped'])

    # Remove self-loops
    links_df = links_df[links_df['protein1_mapped'] != links_df['protein2_mapped']]

    # Remove duplicate edges (keep first)
    links_df = links_df.drop_duplicates(subset=['protein1_mapped', 'protein2_mapped'])

    # Select final columns
    final_links = links_df[['protein1_mapped', 'protein2_mapped', 'combined_score']].copy()
    final_links.columns = ['protein1', 'protein2', 'combined_score']

    return final_links


def process_sequences(sequences_file, id_map):
    """
    Process sequences file and align with mapped IDs
    Returns DataFrame with mapped IDs and sequences
    """
    # Read sequences (FASTA format)
    sequences = {}
    with gzip.open(sequences_file, 'rt') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                # Start new sequence
                current_id = line[1:]  # Remove '>' character
                current_seq = []
            else:
                current_seq.append(line)
        # Save last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)

    # Create DataFrame
    seq_df = pd.DataFrame(list(sequences.items()), columns=['protein_external_id', 'sequence'])

    # Map IDs
    seq_df['protein_id'] = seq_df['protein_external_id'].map(id_map)

    # Remove unmapped entries
    seq_df = seq_df.dropna(subset=['protein_id'])

    # Select final columns
    final_seq = seq_df[['protein_id', 'sequence']].copy()

    return final_seq


def save_aligned_data(nodes_df, edges_df, sequences_df, interim_dir):
    """Save aligned data to interim directory"""
    interim_path = Path(interim_dir)
    interim_path.mkdir(parents=True, exist_ok=True)

    # Save nodes
    nodes_df.to_csv(interim_path / 'nodes.tsv', sep='\t', index=False)

    # Save edges
    edges_df.to_csv(interim_path / 'edges.tsv', sep='\t', index=False)

    # Save sequences as FASTA
    fasta_file = interim_path / 'seqs.fasta'
    with open(fasta_file, 'w') as f:
        for _, row in sequences_df.iterrows():
            f.write(f">{row['protein_id']}\n")
            f.write(f"{row['sequence']}\n")


def main(raw_dir, interim_dir, min_score=700, map_to='ensp', drop_no_sequence=True):
    """Main preprocessing function"""
    raw_path = Path(raw_dir)
    species = '9606'  # Human

    # File paths
    links_file = raw_path / f'{species}.protein.physical.links.v12.0.txt.gz'
    info_file = raw_path / f'{species}.protein.info.v12.0.txt.gz'
    aliases_file = raw_path / f'{species}.protein.aliases.v12.0.txt.gz'
    sequences_file = raw_path / f'{species}.protein.sequences.v12.0.fa.gz'

    # Check if all files exist
    required_files = [links_file, info_file, aliases_file, sequences_file]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    print("Mapping protein IDs...")
    id_map = map_protein_ids(aliases_file, map_to)
    print(f"Mapped {len(id_map)} protein IDs")

    print("Processing links...")
    edges_df = process_links(links_file, id_map, min_score)
    print(f"Filtered and mapped {len(edges_df)} edges")

    print("Processing sequences...")
    sequences_df = process_sequences(sequences_file, id_map)
    print(f"Processed {len(sequences_df)} sequences")

    # Create nodes DataFrame from sequences
    nodes_df = pd.DataFrame({'protein_id': sequences_df['protein_id']})

    # Filter edges to only include nodes with sequences (if requested)
    if drop_no_sequence:
        valid_nodes = set(nodes_df['protein_id'])
        edges_df = edges_df[
            edges_df['protein1'].isin(valid_nodes) &
            edges_df['protein2'].isin(valid_nodes)
        ]
        print(f"Filtered to {len(edges_df)} edges with valid sequences")

    print("Saving aligned data...")
    save_aligned_data(nodes_df, edges_df, sequences_df, interim_dir)
    print(f"Data saved to {interim_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess STRING database files')
    parser.add_argument('--raw_dir', required=True, help='Directory containing raw STRING files')
    parser.add_argument('--interim_dir', required=True, help='Directory to save interim files')
    parser.add_argument('--min_score', type=int, default=700, help='Minimum combined_score to filter links')
    parser.add_argument('--map_to', default='ensp', choices=['ensp', 'preferred_name', 'uniprot'],
                        help='Preferred ID type for mapping')
    parser.add_argument('--drop_no_sequence', action='store_true',
                        help='Drop edges for proteins without sequences')

    args = parser.parse_args()
    main(args.raw_dir, args.interim_dir, args.min_score, args.map_to, args.drop_no_sequence)