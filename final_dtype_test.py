#!/usr/bin/env python3
"""
Final test to ensure all dtype issues are resolved before running on GPU.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_training_pipeline():
    """Test the complete training pipeline for dtype issues."""
    
    print("🧪 FINAL DTYPE COMPATIBILITY TEST")
    print("=" * 60)
    
    try:
        # Test data loading
        from data_loader import DataProcessor
        print("1️⃣ Testing data loading...")
        
        dp = DataProcessor()
        
        # Check critical dtypes
        checks = [
            ("item_degree", dp.item_degree.dtype, torch.float32),
            ("item_popularity_group", dp.item_popularity_group.dtype, torch.int64),
            ("train_edges", dp.train_edges.dtype, torch.int64),
            ("edge_index_train", dp.edge_index_train.dtype, torch.int64),
        ]
        
        for name, actual, expected in checks:
            if actual == expected:
                print(f"   ✅ {name}: {actual}")
            else:
                print(f"   ❌ {name}: {actual} (expected {expected})")
                return False
        
        # Test model creation
        print("2️⃣ Testing model creation...")
        from models import LightGCNRecommender
        import config
        
        model = LightGCNRecommender(
            dp.num_users, dp.num_items, 
            config.EMBEDDING_DIM, config.NUM_LAYERS
        )
        print(f"   ✅ Model created successfully")
        
        # Test batch sampling
        print("3️⃣ Testing batch sampling...")
        from train import sample_bpr_batch
        
        users, pos_items, neg_items = sample_bpr_batch(
            dp.train_edges, dp.num_items, 
            dp.train_user_pos_items, batch_size=16
        )
        
        batch_checks = [
            ("users", users.dtype, torch.int64),
            ("pos_items", pos_items.dtype, torch.int64), 
            ("neg_items", neg_items.dtype, torch.int64),
        ]
        
        for name, actual, expected in batch_checks:
            if actual == expected:
                print(f"   ✅ {name}: {actual}")
            else:
                print(f"   ❌ {name}: {actual} (expected {expected})")
                return False
        
        # Test forward pass
        print("4️⃣ Testing forward pass...")
        device = torch.device('cpu')  # Use CPU for testing
        
        users_dev = users.to(device)
        pos_items_dev = pos_items.to(device)
        neg_items_dev = neg_items.to(device)
        edge_index_dev = dp.edge_index_train.to(device)
        model_dev = model.to(device)
        
        pos_scores, neg_scores, embeddings = model_dev.bpr_forward(
            edge_index_dev, users_dev, pos_items_dev, neg_items_dev
        )
        
        forward_checks = [
            ("pos_scores", pos_scores.dtype, torch.float32),
            ("neg_scores", neg_scores.dtype, torch.float32),
            ("embeddings[0]", embeddings[0].dtype, torch.float32),
        ]
        
        for name, actual, expected in forward_checks:
            if actual == expected:
                print(f"   ✅ {name}: {actual}")
            else:
                print(f"   ❌ {name}: {actual} (expected {expected})")
                return False
        
        # Test loss computation
        print("5️⃣ Testing loss computations...")
        from losses import bpr_loss, l2_regularization
        from ile_losses import ile_loss
        
        # BPR loss
        bpr_loss_val = bpr_loss(pos_scores, neg_scores)
        if bpr_loss_val.dtype != torch.float32:
            print(f"   ❌ bpr_loss: {bpr_loss_val.dtype} (expected torch.float32)")
            return False
        print(f"   ✅ bpr_loss: {bpr_loss_val.dtype}")
        
        # L2 regularization
        u_emb, pos_emb, neg_emb = embeddings
        reg_loss = l2_regularization(u_emb, pos_emb, neg_emb, batch_size=users.size(0))
        if reg_loss.dtype != torch.float32:
            print(f"   ❌ reg_loss: {reg_loss.dtype} (expected torch.float32)")
            return False
        print(f"   ✅ reg_loss: {reg_loss.dtype}")
        
        # ILE loss
        item_groups_dev = dp.item_popularity_group.to(device)
        ile_loss_val = ile_loss(pos_scores, neg_scores, pos_items_dev, item_groups_dev, device)
        if ile_loss_val.dtype != torch.float32:
            print(f"   ❌ ile_loss: {ile_loss_val.dtype} (expected torch.float32)")
            return False
        print(f"   ✅ ile_loss: {ile_loss_val.dtype}")
        
        # Test complete training step
        print("6️⃣ Testing complete training step...")
        
        total_loss = bpr_loss_val + config.WEIGHT_DECAY * reg_loss + 1.0 * ile_loss_val
        
        if total_loss.dtype != torch.float32:
            print(f"   ❌ total_loss: {total_loss.dtype} (expected torch.float32)")
            return False
        print(f"   ✅ total_loss: {total_loss.dtype}")
        
        # Test backward pass
        optimizer = torch.optim.Adam(model_dev.parameters(), lr=0.001)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"   ✅ Backward pass successful")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("   The training pipeline is ready for GPU execution.")
        print("   No dtype errors should occur during training.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_pipeline()
    if success:
        print("\n✅ READY FOR GPU TRAINING!")
        print("   You can now safely run: bash run_on_gpu.sh --train train_all.py")
    else:
        print("\n❌ Still have issues to fix!")
        print("   Please check the errors above.")