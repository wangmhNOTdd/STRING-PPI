#!/usr/bin/env python3
"""
Build graph from processed STRING data and ESM2 features:
1. Load nodes, edges, and ESM2 features
2. Create PyTorch Geometric graph data
3. Perform negative sampling
4. Split data into train/val/test sets
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import networkx as nx


def load_data(nodes_file, edges_file, features_file):
    """Load nodes, edges, and features"""
    # Load nodes
    nodes_df = pd.read_csv(nodes_file, sep='\t')

    # Load edges
    edges_df = pd.read_csv(edges_file, sep='\t')

    # Load features
    features_data = np.load(features_file)
    feature_keys = features_data['keys']
    feature_values = features_data['embeddings']

    # Create feature dictionary
    features = dict(zip(feature_keys, feature_values))

    return nodes_df, edges_df, features


def create_graph(nodes_df, edges_df, features):
    """Create PyTorch Geometric graph data"""
    # Create node ID to index mapping
    node_ids = nodes_df['protein_id'].tolist()
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Create edge index tensor
    edge_index = torch.tensor([
        edges_df['idx1'].tolist() + edges_df['idx2'].tolist(),
        edges_df['idx2'].tolist() + edges_df['idx1'].tolist()  # Add reverse edges for undirected graph
    ], dtype=torch.long)

    # Create feature matrix
    feature_list = []
    for node_id in node_ids:
        if node_id in features:
            feature_list.append(features[node_id])
        else:
            # Use zero vector if no features available
            feature_list.append(np.zeros(features[list(features.keys())[0]].shape))

    x = torch.tensor(np.array(feature_list), dtype=torch.float)

    # Create edge attributes (combined_score)
    edge_attr = torch.tensor(
        edges_df['combined_score'].tolist() + edges_df['combined_score'].tolist(),
        dtype=torch.float
    ).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_id_to_idx


def negative_sampling(edges_df, num_nodes, neg_ratio=1):
    """Perform negative sampling"""
    # Create a set of existing edges for fast lookup
    edge_set = set()
    for _, row in edges_df.iterrows():
        edge_set.add((row['idx1'], row['idx2']))
        edge_set.add((row['idx2'], row['idx1']))  # Add reverse edge

    # Generate negative samples
    neg_edges = []
    num_pos_edges = len(edges_df)
    num_neg_edges = int(num_pos_edges * neg_ratio)

    while len(neg_edges) < num_neg_edges:
        # Randomly sample two nodes
        node1 = np.random.randint(0, num_nodes)
        node2 = np.random.randint(0, num_nodes)

        # Skip if it's a self-loop or already exists
        if node1 == node2 or (node1, node2) in edge_set:
            continue

        # Add negative edge
        neg_edges.append([node1, node2])
        edge_set.add((node1, node2))
        edge_set.add((node2, node1))  # Add reverse edge

    return torch.tensor(neg_edges, dtype=torch.long).t()


def split_data(data, split_type='transductive', val_ratio=0.1, test_ratio=0.2):
    """Split data into train/val/test sets"""
    # Here edge_index contains only positive edges duplicated in both directions.
    num_edges = data.edge_index.size(1) // 2
    edge_indices = list(range(num_edges))

    if split_type == 'transductive':
        # Transductive: all nodes visible, only edges split
        train_idx, temp_idx = train_test_split(edge_indices, test_size=val_ratio+test_ratio, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)

        # Create masks for edges
        train_mask = torch.zeros(num_edges * 2, dtype=torch.bool)
        val_mask = torch.zeros(num_edges * 2, dtype=torch.bool)
        test_mask = torch.zeros(num_edges * 2, dtype=torch.bool)

        # Set masks for both directions of each edge
        for i in train_idx:
            train_mask[i] = True
            train_mask[i + num_edges] = True
        for i in val_idx:
            val_mask[i] = True
            val_mask[i + num_edges] = True
        for i in test_idx:
            test_mask[i] = True
            test_mask[i + num_edges] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    elif split_type == 'inductive':
        # Inductive: split nodes, edges of test nodes go to test set
        num_nodes = data.x.size(0)
        node_indices = list(range(num_nodes))
        train_nodes, temp_nodes = train_test_split(node_indices, test_size=val_ratio+test_ratio, random_state=42)
        val_nodes, test_nodes = train_test_split(temp_nodes, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)

        # Convert to sets for fast lookup
        train_nodes_set = set(train_nodes)
        val_nodes_set = set(val_nodes)
        test_nodes_set = set(test_nodes)

        # Assign edges to splits based on node membership
        train_mask = torch.zeros(num_edges * 2, dtype=torch.bool)
        val_mask = torch.zeros(num_edges * 2, dtype=torch.bool)
        test_mask = torch.zeros(num_edges * 2, dtype=torch.bool)

        for i in range(num_edges):
            node1, node2 = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            if node1 in train_nodes_set and node2 in train_nodes_set:
                train_mask[i] = True
                train_mask[i + num_edges] = True
            elif node1 in val_nodes_set and node2 in val_nodes_set:
                val_mask[i] = True
                val_mask[i + num_edges] = True
            elif node1 in test_nodes_set and node2 in test_nodes_set:
                test_mask[i] = True
                test_mask[i + num_edges] = True
            # Edges between different sets can be assigned to test or ignored

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    elif split_type == 'dfs':
        # DFS-based ordering on the undirected positive edge set, then split by order
        num_nodes = data.x.size(0)
        # Build simple undirected graph from the first half of edges
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edge_list = [(int(data.edge_index[0, i]), int(data.edge_index[1, i])) for i in range(num_edges)]
        G.add_edges_from(edge_list)

        # Compute DFS discovery order for nodes across components
        order = {}
        t = 0
        for start in range(num_nodes):
            if start not in order and G.has_node(start):
                for u in nx.dfs_preorder_nodes(G, source=start):
                    if u not in order:
                        order[u] = t
                        t += 1
        # Fallback for isolated nodes
        for u in range(num_nodes):
            if u not in order:
                order[u] = t
                t += 1

        # Rank edges by min discovery time of endpoints
        ranked = sorted(range(num_edges), key=lambda i: min(order[int(data.edge_index[0, i])], order[int(data.edge_index[1, i])]))

        n_total = len(ranked)
        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = n_total - n_val - n_test

        train_idx = set(ranked[:n_train])
        val_idx = set(ranked[n_train:n_train + n_val])
        test_idx = set(ranked[n_train + n_val:])

        train_mask = torch.zeros(num_edges * 2, dtype=torch.bool)
        val_mask = torch.zeros(num_edges * 2, dtype=torch.bool)
        test_mask = torch.zeros(num_edges * 2, dtype=torch.bool)

        for i in range(num_edges):
            if i in train_idx:
                train_mask[i] = True
                train_mask[i + num_edges] = True
            elif i in val_idx:
                val_mask[i] = True
                val_mask[i + num_edges] = True
            else:
                test_mask[i] = True
                test_mask[i + num_edges] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    return data


def main(nodes_file, edges_file, features_file, neg_ratio=1, split='transductive',
         val_ratio=0.1, test_ratio=0.2, out_dir='data/processed'):
    """Main function to build graph data"""
    print("Loading data...")
    nodes_df, edges_df, features = load_data(nodes_file, edges_file, features_file)
    print(f"Loaded {len(nodes_df)} nodes and {len(edges_df)} edges")

    # Create node ID to index mapping
    node_ids = nodes_df['protein_id'].tolist()
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Filter edges to only include nodes with features
    valid_node_ids = set(node_ids)
    edges_df = edges_df[
        edges_df['protein1'].isin(valid_node_ids) &
        edges_df['protein2'].isin(valid_node_ids)
    ].copy()

    # Map node IDs to indices
    edges_df['idx1'] = edges_df['protein1'].map(node_id_to_idx)
    edges_df['idx2'] = edges_df['protein2'].map(node_id_to_idx)

    print("Creating graph...")
    data, node_id_to_idx = create_graph(nodes_df, edges_df, features)
    print(f"Created graph with {data.x.size(0)} nodes and {data.edge_index.size(1)} edges")

    print("Splitting data...")
    data = split_data(data, split, val_ratio, test_ratio)

    # Save processed data
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.save(data, out_path / 'graph.pt')
    np.save(out_path / 'node_id_to_idx.npy', node_id_to_idx)

    print(f"Graph data saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build graph from STRING data and ESM2 features')
    parser.add_argument('--nodes', required=True, help='Nodes TSV file')
    parser.add_argument('--edges', required=True, help='Edges TSV file')
    parser.add_argument('--features', required=True, help='ESM2 features NPZ file')
    parser.add_argument('--neg_ratio', type=float, default=1, help='Negative sampling ratio')
    parser.add_argument('--split', choices=['transductive', 'inductive', 'dfs'], default='transductive',
                        help='Data split strategy')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--out_dir', default='data/processed', help='Output directory')

    args = parser.parse_args()
    main(args.nodes, args.edges, args.features, args.neg_ratio, args.split,
         args.val_ratio, args.test_ratio, args.out_dir)