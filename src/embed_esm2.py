!/usr/bin/env python3
"""
Generate ESM2 embeddings for protein sequences.
"""

import argparse
import gzip
import os
import torch
import numpy as np
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm

# Check if transformers library is available
try:
    from transformers import AutoTokenizer, EsmModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. ESM2 embeddings cannot be generated.")


def read_fasta_file(fasta_file):
    """Read sequences from a FASTA file"""
    sequences = {}
    with open(fasta_file, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            sequences[record.id] = str(record.seq)
    return sequences


def load_esm2_model(model_name="esm2_t12_35M_UR50D"):
    """Load ESM2 model and tokenizer"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for ESM2 embeddings")

    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
    model = EsmModel.from_pretrained(f"facebook/{model_name}")
    return tokenizer, model


def generate_embeddings(sequences, tokenizer, model, batch_size=1, pooling='mean', device='cpu'):
    """Generate embeddings for sequences"""
    model.to(device)
    model.eval()

    embeddings = {}

    # Convert to list for batching
    seq_ids = list(sequences.keys())

    # Process in batches
    for i in tqdm(range(0, len(seq_ids), batch_size), desc="Generating embeddings"):
        batch_ids = seq_ids[i:i+batch_size]
        batch_seqs = [sequences[seq_id] for seq_id in batch_ids]

        # Tokenize sequences
        # Truncate sequences to maximum effective length of 1022 amino acids
        batch_seqs_truncated = [seq[:1022] for seq in batch_seqs]
        encoded = tokenizer(batch_seqs_truncated, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            hidden_states = outputs.last_hidden_state

            # Apply pooling
            if pooling == 'mean':
                # Mean pooling over sequence length (excluding [CLS] and [SEP] tokens)
                embeddings_batch = hidden_states[:, 1:-1, :].mean(dim=1)
            elif pooling == 'cls':
                # Use [CLS] token embedding
                embeddings_batch = hidden_states[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

        # Store embeddings
        for j, seq_id in enumerate(batch_ids):
            embeddings[seq_id] = embeddings_batch[j].cpu().numpy()

    return embeddings


def save_embeddings(embeddings, output_file):
    """Save embeddings to a .npz file"""
    # Convert to arrays
    keys = list(embeddings.keys())
    values = np.array([embeddings[key] for key in keys])

    # Save
    np.savez_compressed(output_file, keys=keys, embeddings=values)
    print(f"Embeddings saved to {output_file}")


def main(fasta_file, esm2_variant="esm2_t12_35M_UR50D", batch_size=1, pooling="mean",
         device="cpu", output_file="esm2_embeddings.npz"):
    """Main function to generate ESM2 embeddings"""
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping embedding generation.")
        return

    # Read sequences
    print("Reading sequences...")
    sequences = read_fasta_file(fasta_file)
    print(f"Read {len(sequences)} sequences")

    # Load model
    print("Loading ESM2 model...")
    tokenizer, model = load_esm2_model(esm2_variant)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(sequences, tokenizer, model, batch_size, pooling, device)

    # Save embeddings
    print("Saving embeddings...")
    save_embeddings(embeddings, output_file)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ESM2 embeddings for protein sequences")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--esm2_variant", default="esm2_t12_35M_UR50D",
                        help="ESM2 model variant (default: esm2_t12_35M_UR50D)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean",
                        help="Pooling method (default: mean)")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cpu)")
    parser.add_argument("--out_npz", required=True, help="Output .npz file")

    args = parser.parse_args()
    main(args.fasta, args.esm2_variant, args.batch_size, args.pooling, args.device, args.out_npz)