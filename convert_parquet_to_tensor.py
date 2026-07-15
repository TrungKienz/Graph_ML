#!/usr/bin/env python3
"""
Convert parquet data to tensor format for training
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path

def main():
    print("🔄 Converting parquet data to tensor format...")
    
    # Paths
    parquet_dir = Path("preprocess_data/raw_data_after_preprocess_and_split")
    output_dir = Path("preprocess_data")
    output_dir.mkdir(exist_ok=True)
    
    # Load parquet files
    train_df = pd.read_parquet(parquet_dir / "train.parquet")
    val_df = pd.read_parquet(parquet_dir / "val.parquet")
    test_df = pd.read_parquet(parquet_dir / "test.parquet")
    
    print(f"✅ Loaded parquet files:")
    print(f"   Train: {len(train_df)} interactions")
    print(f"   Val: {len(val_df)} interactions") 
    print(f"   Test: {len(test_df)} interactions")
    
    # Extract dimensions
    all_users = set(train_df['user_idx']) | set(val_df['user_idx']) | set(test_df['user_idx'])
    all_items = set(train_df['movie_idx']) | set(val_df['movie_idx']) | set(test_df['movie_idx'])
    
    num_users = max(all_users) + 1
    num_items = max(all_items) + 1
    
    print(f"   Users: {num_users}")
    print(f"   Items: {num_items}")
    
    # Convert to tensors
    train_edges = torch.tensor(train_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    val_edges = torch.tensor(val_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    test_edges = torch.tensor(test_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    
    # Create edge_index for training (bipartite graph)
    user_indices = train_edges[:, 0]
    item_indices = train_edges[:, 1] + num_users  # Offset items
    
    # Bidirectional edges for bipartite graph
    edge_index_train = torch.stack([
        torch.cat([user_indices, item_indices]),
        torch.cat([item_indices, user_indices])
    ])
    
    # Calculate item degrees
    item_degrees = train_df['movie_idx'].value_counts().sort_index()
    item_degree = torch.zeros(num_items, dtype=torch.float)
    for item_id, degree in item_degrees.items():
        item_degree[item_id] = degree
        
    # Create item popularity groups based on degree percentiles
    degree_values = item_degree.float()
    p50 = torch.quantile(degree_values, 0.5)
    p80 = torch.quantile(degree_values, 0.8)
    
    item_popularity_group = torch.zeros(num_items, dtype=torch.long)
    item_popularity_group[degree_values >= p80] = 2  # Head (top 20%)
    item_popularity_group[(degree_values >= p50) & (degree_values < p80)] = 1  # Middle (30%)
    item_popularity_group[degree_values < p50] = 0  # Tail (bottom 50%)
    
    tail_count = (item_popularity_group == 0).sum().item()
    middle_count = (item_popularity_group == 1).sum().item()
    head_count = (item_popularity_group == 2).sum().item()
    
    print(f"📊 Item groups: Tail={tail_count}, Middle={middle_count}, Head={head_count}")
    
    # Save tensors
    torch.save(train_edges, output_dir / "train_edges.pt")
    torch.save(val_edges, output_dir / "val_edges.pt") 
    torch.save(test_edges, output_dir / "test_edges.pt")
    torch.save(edge_index_train, output_dir / "edge_index_train.pt")
    torch.save(item_degree, output_dir / "item_degree.pt")
    torch.save(item_popularity_group, output_dir / "item_popularity_group.pt")
    
    print(f"\n💾 Tensors saved to: {output_dir}")
    print(f"   - train_edges.pt: {train_edges.shape}")
    print(f"   - val_edges.pt: {val_edges.shape}")
    print(f"   - test_edges.pt: {test_edges.shape}")
    print(f"   - edge_index_train.pt: {edge_index_train.shape}")
    print(f"   - item_degree.pt: {item_degree.shape}")
    print(f"   - item_popularity_group.pt: {item_popularity_group.shape}")
    
    print(f"\n✅ Conversion completed! Ready for training.")

if __name__ == "__main__":
    main()