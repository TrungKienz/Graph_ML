#!/usr/bin/env python3
"""
Quick conversion from parquet to tensors
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
    
    # Edge index
    user_indices = train_edges[:, 0]
    item_indices = train_edges[:, 1] + num_users
    edge_index_train = torch.stack([
        torch.cat([user_indices, item_indices]),
        torch.cat([item_indices, user_indices])
    ])
    
    # Item degrees
    item_degrees = train_df['movie_idx'].value_counts().sort_index()
    item_degree = torch.zeros(num_items, dtype=torch.float)  # FIXED: Use float for pow operations
    for item_id, degree in item_degrees.items():
        item_degree[item_id] = float(degree)  # FIXED: Ensure float
    
    # Popularity groups
    degree_values = item_degree.float()
    p50 = torch.quantile(degree_values, 0.5)
    p80 = torch.quantile(degree_values, 0.8)
    
    item_popularity_group = torch.zeros(num_items, dtype=torch.long)
    item_popularity_group[degree_values >= p80] = 2  # Head
    item_popularity_group[(degree_values >= p50) & (degree_values < p80)] = 1  # Middle
    item_popularity_group[degree_values < p50] = 0  # Tail
    
    print("Saving tensors...")
    torch.save(train_edges, output_dir / "train_edges.pt")
    torch.save(val_edges, output_dir / "val_edges.pt")
    torch.save(test_edges, output_dir / "test_edges.pt")
    torch.save(edge_index_train, output_dir / "edge_index_train.pt")
    torch.save(item_degree, output_dir / "item_degree.pt")
    torch.save(item_popularity_group, output_dir / "item_popularity_group.pt")
    
    print("Done! Files saved to preprocess_data/")

if __name__ == "__main__":
    main()