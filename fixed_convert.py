#!/usr/bin/env python3
"""
FIXED conversion from parquet to tensors - Correct edge_index construction
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path

def main():
    # Paths
    parquet_dir = Path("preprocess_data/raw_data_after_preprocess_and_split")
    output_dir = Path("preprocess_data")
    
    print("Loading parquet files...")
    train_df = pd.read_parquet(parquet_dir / "train.parquet")
    val_df = pd.read_parquet(parquet_dir / "val.parquet")
    test_df = pd.read_parquet(parquet_dir / "test.parquet")
    
    print("Converting to tensors...")
    
    # Edges
    train_edges = torch.tensor(train_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    val_edges = torch.tensor(val_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    test_edges = torch.tensor(test_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    
    num_users = 6034
    num_items = 3533
    
    print(f"Data info:")
    print(f"  Users: 0-{num_users-1}")
    print(f"  Items: 0-{num_items-1}")
    print(f"  Train edges: {len(train_edges)}")
    
    # FIXED: Correct edge_index construction for bipartite graph
    # In bipartite graph: users are nodes 0..num_users-1, items are nodes num_users..num_users+num_items-1
    user_nodes = train_edges[:, 0]  # User nodes: 0..6033
    item_nodes = train_edges[:, 1] + num_users  # Item nodes: 6034..9566
    
    # Create bidirectional edges: user->item AND item->user
    edge_index_train = torch.stack([
        torch.cat([user_nodes, item_nodes]),  # Sources: users + items
        torch.cat([item_nodes, user_nodes])   # Targets: items + users  
    ])
    
    print(f"Edge index shape: {edge_index_train.shape}")
    print(f"Edge index range: [{edge_index_train.min()}, {edge_index_train.max()}]")
    print(f"Expected range: [0, {num_users + num_items - 1}]")
    
    # Verify no out-of-bounds indices
    max_node = num_users + num_items - 1
    if edge_index_train.max() > max_node:
        print(f"❌ ERROR: Edge index out of bounds! Max: {edge_index_train.max()}, Expected max: {max_node}")
        return
    
    # Item degrees
    item_degrees = train_df['movie_idx'].value_counts().sort_index()
    item_degree = torch.zeros(num_items, dtype=torch.float)  # FIXED: Use float for pow operations
    for item_id, degree in item_degrees.items():
        if item_id < num_items:  # Safety check
            item_degree[item_id] = float(degree)  # FIXED: Ensure float
    
    print(f"Item degree range: [{item_degree.min()}, {item_degree.max()}]")
    
    # Popularity groups
    degree_values = item_degree.float()
    p50 = torch.quantile(degree_values, 0.5)
    p80 = torch.quantile(degree_values, 0.8)
    
    item_popularity_group = torch.zeros(num_items, dtype=torch.long)
    item_popularity_group[degree_values >= p80] = 2  # Head
    item_popularity_group[(degree_values >= p50) & (degree_values < p80)] = 1  # Middle
    item_popularity_group[degree_values < p50] = 0  # Tail
    
    tail_count = (item_popularity_group == 0).sum().item()
    middle_count = (item_popularity_group == 1).sum().item()
    head_count = (item_popularity_group == 2).sum().item()
    
    print(f"Item groups: Tail={tail_count}, Middle={middle_count}, Head={head_count}")
    
    # Verify test: extract item indices from edge_index and check bounds
    print("\n🧪 Testing edge_index bounds...")
    test_item_indices = edge_index_train[1] - num_users
    valid_mask = (test_item_indices >= 0) & (test_item_indices < num_items)
    invalid_count = (~valid_mask).sum().item()
    
    if invalid_count > 0:
        print(f"❌ Found {invalid_count} invalid item indices after offset!")
        print(f"  Min item index: {test_item_indices.min()}")
        print(f"  Max item index: {test_item_indices.max()}")
        print(f"  Expected range: [0, {num_items-1}]")
        return
    else:
        print("✅ All item indices are valid after offset")
    
    print("\nSaving tensors...")
    torch.save(train_edges, output_dir / "train_edges.pt")
    torch.save(val_edges, output_dir / "val_edges.pt")
    torch.save(test_edges, output_dir / "test_edges.pt")
    torch.save(edge_index_train, output_dir / "edge_index_train.pt")
    torch.save(item_degree, output_dir / "item_degree.pt")
    torch.save(item_popularity_group, output_dir / "item_popularity_group.pt")
    
    print("✅ Done! Fixed tensor files saved to preprocess_data/")

if __name__ == "__main__":
    main()