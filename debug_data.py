#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.data_loader import DataProcessor

def main():
    print("Loading data...")
    data_processor = DataProcessor()
    
    print(f"Users: {data_processor.num_users}")
    print(f"Items: {data_processor.num_items}")
    print(f"Edge index shape: {data_processor.edge_index_train.shape}")
    
    # Check edge_index bounds  
    max_node = data_processor.num_users + data_processor.num_items - 1
    edge_min = data_processor.edge_index_train.min()
    edge_max = data_processor.edge_index_train.max()
    
    print(f"Edge range: [{edge_min}, {edge_max}]")
    print(f"Expected max: {max_node}")
    
    if edge_max > max_node:
        print("❌ EDGE INDEX OUT OF BOUNDS!")
    else:
        print("✅ Edge indices OK")
        
    # Test item index extraction (this is what causes CUDA error)
    item_indices = data_processor.edge_index_train[1] - data_processor.num_users
    item_min = item_indices.min()
    item_max = item_indices.max()
    
    print(f"Item indices after offset: [{item_min}, {item_max}]")
    print(f"Expected range: [0, {data_processor.num_items-1}]")
    
    if item_min < 0 or item_max >= data_processor.num_items:
        print("❌ ITEM INDICES OUT OF BOUNDS - THIS CAUSES CUDA ERROR!")
    else:
        print("✅ Item indices OK")

if __name__ == "__main__":
    main()