#!/usr/bin/env python3
"""
Debug script to check and fix all dtype issues in the training pipeline.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append('src')

# Import modules
from data_loader import DataProcessor
from models import LightGCNRecommender
from train import sample_bpr_batch
import config

def debug_dtype_issues():
    """Debug and verify all dtype issues are resolved."""
    
    print("🔍 DEBUGGING DTYPE ISSUES")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load data and check dtypes
    print("\n1️⃣ CHECKING DATA DTYPES:")
    data_processor = DataProcessor()
    
    print(f"   item_degree dtype: {data_processor.item_degree.dtype}")
    print(f"   item_popularity_group dtype: {data_processor.item_popularity_group.dtype}")
    print(f"   train_edges dtype: {data_processor.train_edges.dtype}")
    print(f"   edge_index_train dtype: {data_processor.edge_index_train.dtype}")
    
    # Verify item_degree is float
    if data_processor.item_degree.dtype != torch.float32:
        print(f"❌ FIXING item_degree dtype: {data_processor.item_degree.dtype} -> torch.float32")
        data_processor.item_degree = data_processor.item_degree.float()
    else:
        print(f"✅ item_degree dtype is correct: {data_processor.item_degree.dtype}")
    
    # 2. Test BPR batch sampling
    print("\n2️⃣ CHECKING BPR BATCH SAMPLING:")
    users, pos_items, neg_items = sample_bpr_batch(
        data_processor.train_edges,
        data_processor.num_items, 
        data_processor.train_user_pos_items,
        batch_size=32
    )
    
    print(f"   users dtype: {users.dtype}")
    print(f"   pos_items dtype: {pos_items.dtype}")
    print(f"   neg_items dtype: {neg_items.dtype}")
    
    # All should be torch.long for indexing
    assert users.dtype == torch.long, f"users dtype should be long, got {users.dtype}"
    assert pos_items.dtype == torch.long, f"pos_items dtype should be long, got {pos_items.dtype}"
    assert neg_items.dtype == torch.long, f"neg_items dtype should be long, got {neg_items.dtype}"
    print("✅ BPR batch dtypes are correct")
    
    # 3. Test model creation and forward pass
    print("\n3️⃣ CHECKING MODEL FORWARD PASS:")
    model = LightGCNRecommender(
        data_processor.num_users,
        data_processor.num_items, 
        config.EMBEDDING_DIM,
        config.NUM_LAYERS
    ).to(device)
    
    # Move data to device
    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)
    edge_index = data_processor.edge_index_train.to(device)
    
    try:
        pos_scores, neg_scores, embeddings = model.bpr_forward(
            edge_index, users, pos_items, neg_items
        )
        
        print(f"   pos_scores dtype: {pos_scores.dtype}")
        print(f"   neg_scores dtype: {neg_scores.dtype}")
        print(f"   embeddings[0] dtype: {embeddings[0].dtype}")
        print(f"   embeddings[1] dtype: {embeddings[1].dtype}")
        print(f"   embeddings[2] dtype: {embeddings[2].dtype}")
        
        # All should be float32
        assert pos_scores.dtype == torch.float32, f"pos_scores should be float32, got {pos_scores.dtype}"
        assert neg_scores.dtype == torch.float32, f"neg_scores should be float32, got {neg_scores.dtype}"
        print("✅ Model forward pass dtypes are correct")
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False
    
    # 4. Test loss computations
    print("\n4️⃣ CHECKING LOSS COMPUTATIONS:")
    
    from losses import bpr_loss, l2_regularization
    from ile_losses import ile_loss
    
    try:
        # BPR loss
        bpr_loss_val = bpr_loss(pos_scores, neg_scores)
        print(f"   bpr_loss dtype: {bpr_loss_val.dtype}")
        assert bpr_loss_val.dtype == torch.float32, f"bpr_loss should be float32, got {bpr_loss_val.dtype}"
        
        # L2 regularization
        u_emb, pos_emb, neg_emb = embeddings
        reg_loss = l2_regularization(u_emb, pos_emb, neg_emb, batch_size=users.size(0))
        print(f"   reg_loss dtype: {reg_loss.dtype}")
        assert reg_loss.dtype == torch.float32, f"reg_loss should be float32, got {reg_loss.dtype}"
        
        # ILE loss
        item_groups = data_processor.item_popularity_group.to(device)
        ile_loss_val = ile_loss(pos_scores, neg_scores, pos_items, item_groups, device)
        print(f"   ile_loss dtype: {ile_loss_val.dtype}")
        assert ile_loss_val.dtype == torch.float32, f"ile_loss should be float32, got {ile_loss_val.dtype}"
        
        print("✅ Loss computations dtypes are correct")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test complete training step
    print("\n5️⃣ TESTING COMPLETE TRAINING STEP:")
    
    try:
        # Total loss
        lambda_ile = 1.0
        total_loss = (bpr_loss_val + 
                     config.WEIGHT_DECAY * reg_loss + 
                     lambda_ile * ile_loss_val)
        
        print(f"   total_loss dtype: {total_loss.dtype}")
        print(f"   total_loss value: {total_loss.item():.6f}")
        assert total_loss.dtype == torch.float32, f"total_loss should be float32, got {total_loss.dtype}"
        assert torch.isfinite(total_loss), f"total_loss should be finite, got {total_loss}"
        
        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"   gradient norm: {grad_norm.item():.6f}")
        
        optimizer.step()
        
        print("✅ Complete training step successful")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL DTYPE CHECKS PASSED!")
    print("   The training pipeline should now work without int dtype errors.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = debug_dtype_issues()
    if success:
        print("\n✅ Ready to run training!")
    else:
        print("\n❌ Still have issues to fix!")