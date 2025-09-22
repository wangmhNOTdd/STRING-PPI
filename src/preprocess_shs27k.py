#!/usr/bin/env python3
"""
Preprocess SHS27K dataset into nodes/edges/seqs compatible with build_graph.py

Inputs (under data/SHS27K/raw/ by default):
- protein.actions.SHS27k.STRING.txt  (TSV with header; columns include item_id_a,item_id_b,score,...)
- protein.SHS27k.sequences.dictionary.tsv (TSV without header: protein_id<tab>sequence)

Outputs (under data/SHS27K/interim/ by default):
- nodes.tsv              (columns: protein_id)
- edges.tsv              (columns: protein1, protein2, combined_score)
- seqs.fasta             (FASTA with >protein_id and sequence)

Notes:
- Deduplicates multi-rows per pair by taking the max score, treats graph as undirected.
- Drops self-loops and edges with missing sequences when --drop_no_sequence is set.
"""

import argparse
from pathlib import Path
import pandas as pd


def read_actions(actions_file: Path) -> pd.DataFrame:
    df = pd.read_csv(actions_file, sep='\t', header=0)
    # keep essential columns and coerce score to int
    need_cols = ['item_id_a', 'item_id_b', 'score']
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {actions_file}")
    df = df[need_cols].copy()
    df = df.dropna(subset=['item_id_a', 'item_id_b', 'score'])
    df['score'] = df['score'].astype(int)
    # drop self-loops
    df = df[df['item_id_a'] != df['item_id_b']]
    # undirected de-dup: sort endpoints, then take max score
    a = df['item_id_a']
    b = df['item_id_b']
    pair_min = a.where(a < b, b)
    pair_max = b.where(a < b, a)
    df = pd.DataFrame({
        'u': pair_min,
        'v': pair_max,
        'score': df['score']
    })
    agg = df.groupby(['u', 'v'], as_index=False)['score'].max()
    agg = agg.rename(columns={'u': 'protein1', 'v': 'protein2', 'score': 'combined_score'})
    return agg


def read_seq_dict(seq_dict_file: Path) -> pd.DataFrame:
    # No header: two columns -> protein_id, sequence
    seq_df = pd.read_csv(seq_dict_file, sep='\t', header=None, names=['protein_id', 'sequence'])
    seq_df = seq_df.dropna(subset=['protein_id', 'sequence'])
    # keep first occurrence if duplicates
    seq_df = seq_df.drop_duplicates(subset=['protein_id'], keep='first').reset_index(drop=True)
    return seq_df


def save_outputs(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, seq_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # nodes.tsv
    nodes_df[['protein_id']].to_csv(out_dir / 'nodes.tsv', sep='\t', index=False)
    # edges.tsv
    edges_df[['protein1', 'protein2', 'combined_score']].to_csv(out_dir / 'edges.tsv', sep='\t', index=False)
    # seqs.fasta
    fasta_path = out_dir / 'seqs.fasta'
    with open(fasta_path, 'w') as f:
        for _, row in seq_df.iterrows():
            f.write(f">{row['protein_id']}\n{row['sequence']}\n")


def main(actions_file: str,
         seqs_dict_file: str,
         interim_dir: str,
         min_score: int | None = 700,
         drop_no_sequence: bool = True,
         no_score_filter: bool = False):
    actions_path = Path(actions_file)
    seqs_path = Path(seqs_dict_file)
    interim_path = Path(interim_dir)

    print("Reading actions...")
    edges_df = read_actions(actions_path)
    print(f"Loaded {len(edges_df)} unique undirected pairs")

    if not no_score_filter and min_score is not None:
        before = len(edges_df)
        edges_df = edges_df[edges_df['combined_score'] >= min_score].reset_index(drop=True)
        print(f"Filtered by combined_score >= {min_score}: {before} -> {len(edges_df)}")
    else:
        print("No score filtering applied (using all edges)")

    print("Reading sequence dictionary...")
    seq_df = read_seq_dict(seqs_path)
    print(f"Loaded {len(seq_df)} sequences")

    if drop_no_sequence:
        print("Aligning edges to available sequences...")
        valid = set(seq_df['protein_id'].tolist())
        before = len(edges_df)
        edges_df = edges_df[
            edges_df['protein1'].isin(valid) & edges_df['protein2'].isin(valid)
        ].reset_index(drop=True)
        print(f"Kept edges with both sequences present: {before} -> {len(edges_df)}")

    # nodes are proteins present in sequences and appearing in edges
    used_nodes = pd.unique(pd.concat([edges_df['protein1'], edges_df['protein2']], ignore_index=True))
    nodes_df = pd.DataFrame({'protein_id': used_nodes})
    seq_df = seq_df[seq_df['protein_id'].isin(used_nodes)].reset_index(drop=True)

    print("Saving outputs...")
    save_outputs(nodes_df, edges_df, seq_df, interim_path)
    print(f"Saved to {interim_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess SHS27K dataset into aligned nodes/edges/seqs')
    parser.add_argument('--actions_file', default='data/SHS27K/raw/protein.actions.SHS27k.STRING.txt',
                        help='Path to SHS27K actions file (TSV)')
    parser.add_argument('--seqs_dict_file', default='data/SHS27K/raw/protein.SHS27k.sequences.dictionary.tsv',
        help='Path to SHS27K sequences dictionary (TSV, no header)')
    parser.add_argument('--interim_dir', default='data/SHS27K/interim', help='Directory to write outputs')
    parser.add_argument('--min_score', type=int, default=700, help='Minimum combined_score threshold (ignored if --no_score_filter)')
    parser.add_argument('--drop_no_sequence', action='store_true', help='Drop edges without both sequences')
    parser.add_argument('--no_score_filter', action='store_true', help='Do not filter by combined_score')

    args = parser.parse_args()
    main(args.actions_file, args.seqs_dict_file, args.interim_dir, args.min_score, args.drop_no_sequence, args.no_score_filter)
