#!/usr/bin/env python3
"""
Quick test to verify ILE fix
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataProcessor
from src.ile_training import train_model_with_ile
import torch

def test_ile_minimal():
    """Test minimal ILE training to verify fix."""
    
    print("🧪 Testing ILE fix with minimal training...")
    
    try:
        # Load data
        print("📁 Loading data...")
        data_processor = DataProcessor()
        
        print(f"✅ Data loaded: {data_processor.num_users} users, {data_processor.num_items} items")
        print(f"📊 Item degree dtype: {data_processor.item_degree.dtype}")
        
        # Try minimal training (2 epochs)
        print("🚀 Starting minimal training (2 epochs)...")
        
        # Override config temporarily
        import src.config as config
        original_epochs = config.NUM_EPOCHS
        config.NUM_EPOCHS = 2  # Just 2 epochs for test
        
        model, metrics = train_model_with_ile(
            train_edges=data_processor.train_edges,
            edge_index_train=data_processor.edge_index_train,
            val_edges=data_processor.val_edges,
            test_edges=data_processor.test_edges,
            num_users=data_processor.num_users,
            num_items=data_processor.num_items,
            train_user_pos_items=data_processor.train_user_pos_items,
            val_user_pos_items=data_processor.val_user_pos_items,
            test_user_pos_items=data_processor.test_user_pos_items,
            item_degree=data_processor.item_degree,
            item_popularity_group=data_processor.item_popularity_group,
            lambda_ile=0.1,
            model_name="test_ile_fix",
            save_checkpoints=False
        )
        
        # Restore config
        config.NUM_EPOCHS = original_epochs
        
        print("✅ SUCCESS! ILE training completed without 'int' pow() errors")
        print(f"📊 Test results: Recall@20={metrics.get('test_Recall@20', 0.0):.4f}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ile_minimal()
    sys.exit(0 if success else 1)