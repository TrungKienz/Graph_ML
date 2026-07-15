#!/usr/bin/env python3
"""
Generate correct tensor files using DataProcessor conversion logic
"""
import sys
from pathlib import Path

# Add src to Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🔧 Generating correct tensor files...")
    
    # This will trigger _convert_and_load_parquet since tensor files don't exist
    from src.data_loader import DataProcessor
    
    try:
        data_processor = DataProcessor()
        print("✅ Tensor files generated successfully!")
        
        # Verify data integrity
        print(f"\n📊 Data verification:")
        print(f"Users: {data_processor.num_users}")
        print(f"Items: {data_processor.num_items}")
        print(f"Edge index shape: {data_processor.edge_index_train.shape}")
        
        # Check bounds
        max_node = data_processor.num_users + data_processor.num_items - 1
        edge_max = data_processor.edge_index_train.max()
        
        if edge_max <= max_node:
            print("✅ Edge indices are within bounds")
        else:
            print(f"❌ Edge indices out of bounds: max={edge_max}, expected_max={max_node}")
            
        # Check item index extraction
        item_indices = data_processor.edge_index_train[1] - data_processor.num_users
        item_min, item_max = item_indices.min(), item_indices.max()
        
        if item_min >= 0 and item_max < data_processor.num_items:
            print("✅ Item indices after offset are valid")
        else:
            print(f"❌ Item indices invalid: range=[{item_min}, {item_max}], expected=[0, {data_processor.num_items-1}]")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready for training!")
    else:
        print("\n💥 Fix needed before training")