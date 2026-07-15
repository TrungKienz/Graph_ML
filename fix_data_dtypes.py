#!/usr/bin/env python3
"""
Fix all data dtype issues by regenerating tensors with correct dtypes.
This should be run on the server to ensure consistent data.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('src')
import config

def fix_and_regenerate_data():
    """Regenerate all data tensors with correct dtypes."""
    
    print("🔧 FIXING DATA DTYPES - COMPLETE REGENERATION")
    print("=" * 70)
    
    # Paths
    parquet_dir = Path("preprocess_data/raw_data_after_preprocess_and_split")
    tensor_dir = Path("preprocess_data")
    tensor_dir.mkdir(exist_ok=True)
    
    print(f"📁 Loading from: {parquet_dir}")
    
    # 1. Load parquet files
    try:
        train_df = pd.read_parquet(parquet_dir / "train.parquet")
        val_df = pd.read_parquet(parquet_dir / "val.parquet")
        test_df = pd.read_parquet(parquet_dir / "test.parquet")
        
        print(f"✅ Loaded parquet files:")
        print(f"   Train: {len(train_df)} interactions")
        print(f"   Val: {len(val_df)} interactions")
        print(f"   Test: {len(test_df)} interactions")
        
    except Exception as e:
        print(f"❌ Failed to load parquet files: {e}")
        return False
    
    # 2. Extract dimensions
    num_users = max(train_df['user_idx'].max(), val_df['user_idx'].max(), test_df['user_idx'].max()) + 1
    num_items = max(train_df['movie_idx'].max(), val_df['movie_idx'].max(), test_df['movie_idx'].max()) + 1
    
    print(f"📊 Dimensions:")
    print(f"   Users: {num_users} (0-{num_users-1})")
    print(f"   Items: {num_items} (0-{num_items-1})")
    
    # 3. Create edge tensors (LONG dtype for indexing)
    print("\n🔗 Creating edge tensors...")
    train_edges = torch.tensor(train_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    val_edges = torch.tensor(val_df[['user_idx', 'movie_idx']].values, dtype=torch.long)  
    test_edges = torch.tensor(test_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    
    print(f"   train_edges: {train_edges.shape}, dtype={train_edges.dtype}")
    print(f"   val_edges: {val_edges.shape}, dtype={val_edges.dtype}")
    print(f"   test_edges: {test_edges.shape}, dtype={test_edges.dtype}")
    
    # 4. Create edge_index for bipartite graph (LONG dtype for indexing)
    print("\n🌐 Creating bipartite graph...")
    user_nodes = train_edges[:, 0]  # Users: 0..num_users-1
    item_nodes = train_edges[:, 1] + num_users  # Items: num_users..num_users+num_items-1
    
    edge_index_train = torch.stack([
        torch.cat([user_nodes, item_nodes]),  # Sources
        torch.cat([item_nodes, user_nodes])   # Targets
    ])
    
    print(f"   edge_index_train: {edge_index_train.shape}, dtype={edge_index_train.dtype}")
    print(f"   Node range: [{edge_index_train.min()}, {edge_index_train.max()}]")
    
    # 5. Create item degrees (FLOAT dtype for calculations)
    print("\n📈 Creating item degrees...")
    item_degrees_series = train_df['movie_idx'].value_counts().sort_index()
    
    # CRITICAL: Create as FLOAT tensor explicitly
    item_degree = torch.zeros(num_items, dtype=torch.float32)
    for item_id, degree in item_degrees_series.items():
        if 0 <= item_id < num_items:
            item_degree[item_id] = float(degree)  # Explicit float conversion
    
    print(f"   item_degree: {item_degree.shape}, dtype={item_degree.dtype}")
    print(f"   Degree range: [{item_degree.min():.1f}, {item_degree.max():.1f}]")
    print(f"   Mean degree: {item_degree.mean():.2f}")
    
    # Verify no integers in item_degree
    assert item_degree.dtype == torch.float32, f"item_degree must be float32, got {item_degree.dtype}"
    
    # 6. Create item popularity groups (LONG dtype for indexing)
    print("\n🏷️ Creating item popularity groups...")
    
    # Use float degrees for percentile calculations
    degree_values = item_degree.float()
    
    if degree_values.sum() == 0:
        item_popularity_group = torch.zeros(num_items, dtype=torch.long)
        print("⚠️  All items have zero degree, assigning all to tail group")
    else:
        # Calculate percentiles on non-zero degrees
        nonzero_degrees = degree_values[degree_values > 0]
        if len(nonzero_degrees) > 0:
            p50 = torch.quantile(nonzero_degrees, 0.5).item()
            p80 = torch.quantile(nonzero_degrees, 0.8).item()
        else:
            p50 = p80 = 0
        
        item_popularity_group = torch.zeros(num_items, dtype=torch.long)
        item_popularity_group[degree_values >= p80] = config.GROUP_HEAD      # Head (top 20%)
        item_popularity_group[(degree_values >= p50) & (degree_values < p80)] = config.GROUP_MIDDLE  # Middle (30%)
        item_popularity_group[degree_values < p50] = config.GROUP_TAIL       # Tail (bottom 50%)
    
    tail_count = (item_popularity_group == config.GROUP_TAIL).sum().item()
    middle_count = (item_popularity_group == config.GROUP_MIDDLE).sum().item()
    head_count = (item_popularity_group == config.GROUP_HEAD).sum().item()
    
    print(f"   item_popularity_group: {item_popularity_group.shape}, dtype={item_popularity_group.dtype}")
    print(f"   Groups: Tail={tail_count}, Middle={middle_count}, Head={head_count}")
    
    # 7. Save all tensors
    print("\n💾 Saving tensors...")
    
    tensors_to_save = {
        'train_edges.pt': train_edges,
        'val_edges.pt': val_edges,  
        'test_edges.pt': test_edges,
        'edge_index_train.pt': edge_index_train,
        'item_degree.pt': item_degree,  # This is the critical one - must be float32
        'item_popularity_group.pt': item_popularity_group,
    }
    
    for filename, tensor in tensors_to_save.items():
        filepath = tensor_dir / filename
        torch.save(tensor, filepath)
        print(f"   ✅ Saved {filename}: {tensor.shape}, {tensor.dtype}")
    
    # 8. Save metadata
    metadata = {
        'num_users': num_users,
        'num_items': num_items,
        'num_train_edges': len(train_edges),
        'num_val_edges': len(val_edges),
        'num_test_edges': len(test_edges),
    }
    
    torch.save(metadata, tensor_dir / 'metadata.pt')
    print(f"   ✅ Saved metadata.pt")
    
    # 9. Verification
    print("\n🧪 VERIFICATION:")
    
    # Load and verify dtypes
    loaded_item_degree = torch.load(tensor_dir / 'item_degree.pt')
    loaded_groups = torch.load(tensor_dir / 'item_popularity_group.pt')
    loaded_edges = torch.load(tensor_dir / 'train_edges.pt')
    
    print(f"   ✅ item_degree: {loaded_item_degree.dtype} (should be torch.float32)")
    print(f"   ✅ item_popularity_group: {loaded_groups.dtype} (should be torch.int64)")  
    print(f"   ✅ train_edges: {loaded_edges.dtype} (should be torch.int64)")
    
    assert loaded_item_degree.dtype == torch.float32, f"FAILED: item_degree dtype is {loaded_item_degree.dtype}"
    assert loaded_groups.dtype == torch.long, f"FAILED: item_popularity_group dtype is {loaded_groups.dtype}"
    assert loaded_edges.dtype == torch.long, f"FAILED: train_edges dtype is {loaded_edges.dtype}"
    
    print("\n" + "=" * 70)
    print("🎉 DATA DTYPE FIXES COMPLETED SUCCESSFULLY!")
    print("   All tensors have been regenerated with correct dtypes:")
    print(f"   - item_degree: torch.float32 (for math operations)")
    print(f"   - edges/indices: torch.long (for indexing)")
    print("   The training should now work without dtype errors.")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = fix_and_regenerate_data()
    if success:
        print("\n✅ Data ready! You can now run training.")
        print("   Recommend: rm preprocess_data/tensor_cache/* (if exists)")
        print("   Then run: python train_all.py")
    else:
        print("\n❌ Failed to fix data!")