#!/usr/bin/env python3
"""
Train HGCN model for PPI prediction:
1. Load graph data
2. Initialize HGCN model
3. Train model with BCE loss
4. Evaluate on validation and test sets
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
import yaml
import numpy as np
from pathlib import Path
import os

# Try to import model components
try:
    # Assuming the models are in the models/ directory
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

    # Import model components (these would need to be implemented)
    # from models.layers.hgcn import HGCN
    # from models.manifolds.poincare import PoincareBall
    # from models.decoders.hyp_distance import HypDistanceDecoder
    pass
except ImportError:
    print("Warning: Model components not available. Using placeholder implementations.")


class PlaceholderHGCN(nn.Module):
    """Placeholder HGCN model"""
    def __init__(self, in_dim, out_dim, manifold=None):
        super().__init__()
        self.manifold = manifold
        self.linear1 = nn.Linear(in_dim, 128)
        self.linear2 = nn.Linear(128, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class PlaceholderDecoder(nn.Module):
    """Placeholder decoder that computes dot product similarity"""
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index):
        # Simple dot product for similarity
        row, col = edge_index
        similarity = torch.sum(z[row] * z[col], dim=1)
        return torch.sigmoid(similarity)


def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def train_model(model, decoder, data, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    decoder.train()

    optimizer.zero_grad()

    # Forward pass
    z = model(data.x.to(device), data.edge_index.to(device))
    pos_pred = decoder(z, data.edge_index.to(device))

    # For negative sampling, we need to create negative edges
    neg_edge_index = negative_sampling(
        data.edge_index,
        num_nodes=data.x.size(0),
        num_neg_samples=data.edge_index.size(1) // 2  # Same number as positive samples
    ).to(device)

    neg_pred = decoder(z, neg_edge_index)

    # Combine positive and negative predictions
    pred = torch.cat([pos_pred, neg_pred])
    labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

    # Compute loss
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_model(model, decoder, data, criterion, device, mask_name):
    """Evaluate model on a specific mask"""
    model.eval()
    decoder.eval()

    with torch.no_grad():
        # Forward pass
        z = model(data.x.to(device), data.edge_index.to(device))

        # Get edges for evaluation
        if mask_name == 'val':
            edge_mask = data.val_mask
        elif mask_name == 'test':
            edge_mask = data.test_mask
        else:
            edge_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)

        edge_index = data.edge_index[:, edge_mask].to(device)

        if edge_index.size(1) == 0:
            return float('inf'), 0.0

        pos_pred = decoder(z, edge_index)

        # For negative sampling
        neg_edge_index = negative_sampling(
            edge_index,
            num_nodes=data.x.size(0),
            num_neg_samples=edge_index.size(1)
        ).to(device)

        neg_pred = decoder(z, neg_edge_index)

        # Combine positive and negative predictions
        pred = torch.cat([pos_pred, neg_pred])
        labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        # Compute loss and accuracy
        loss = criterion(pred, labels)
        acc = ((pred > 0.5) == labels).float().mean()

    return loss.item(), acc.item()


def main(config_file, data_dir, epochs=100, lr=0.001, batch_size=8192,
         model_type='hgcn', decoder_type='hyp_distance', log_dir='runs'):
    """Main training function"""
    # Load configuration
    config = load_config(config_file) if config_file else {}

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = Path(data_dir) / 'graph.pt'
    data = torch.load(data_path, weights_only=False)
    print(f"Loaded graph data with {data.x.size(0)} nodes and {data.edge_index.size(1)} edges")

    # Initialize model
    in_dim = data.x.size(1)
    out_dim = config.get('dim', 128)

    if model_type == 'hgcn':
        model = PlaceholderHGCN(in_dim, out_dim)
    else:
        model = PlaceholderHGCN(in_dim, out_dim)

    model = model.to(device)

    # Initialize decoder
    if decoder_type == 'hyp_distance':
        decoder = PlaceholderDecoder()
    else:
        decoder = PlaceholderDecoder()

    decoder = decoder.to(device)

    # Initialize optimizer and loss
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.BCELoss()

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    best_ckpt_path = log_path / 'best.pt'

    print("Starting training...")
    for epoch in range(epochs):
        # Train
        train_loss = train_model(model, decoder, data, optimizer, criterion, device)

        # Evaluate
        val_loss, val_acc = evaluate_model(model, decoder, data, criterion, device, 'val')
        test_loss, test_acc = evaluate_model(model, decoder, data, criterion, device, 'test')

        print(f"Epoch {epoch+1:03d}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, best_ckpt_path)
            print(f"  -> Saved new best model with val_loss: {val_loss:.4f}")

    print("Training completed!")
    print(f"Best model saved to {best_ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HGCN model for PPI prediction')
    parser.add_argument('--config', help='Model configuration YAML file')
    parser.add_argument('--data_dir', required=True, help='Directory containing graph data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size')
    parser.add_argument('--model', default='hgcn', help='Model type')
    parser.add_argument('--decoder', default='hyp_distance', help='Decoder type')
    parser.add_argument('--log_dir', default='runs', help='Logging directory')

    args = parser.parse_args()
    main(args.config, args.data_dir, args.epochs, args.lr, args.batch_size,
         args.model, args.decoder, args.log_dir)