#!/usr/bin/env python3
"""
Minimal test to check basic training functionality
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🧪 Minimal training test...")
    
    try:
        from src.data_loader import DataProcessor
        from src.ile_training import train_model_with_ile
        
        print("📁 Loading data...")
        data_processor = DataProcessor()
        
        print("🚀 Testing training (1 epoch)...")
        
        # Override epochs for quick test
        import src.config as config
        config.NUM_EPOCHS = 1
        
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
            lambda_ile=0.0,
            model_name="test_model",
            save_checkpoints=False,
            dropout_type=None
        )
        
        print("✅ SUCCESS: Training completed!")
        print(f"📊 Test metrics: {metrics.get('val_Recall@20', 'N/A')}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}")