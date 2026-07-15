#!/usr/bin/env python3
"""
Quick test to verify data loading from new parquet format
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataProcessor
from src import config

def main():
    print("🧪 Testing data loading...")
    
    # Load data
    data_processor = DataProcessor()
    
    # Print statistics
    stats = data_processor.get_statistics()
    
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Data loading test completed!")
    print(f"   Ready for training with {stats['num_users']} users and {stats['num_items']} items")

if __name__ == "__main__":
    main()