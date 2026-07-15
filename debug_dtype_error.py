#!/usr/bin/env python3
"""
Debug the 'int' object has no attribute 'dtype' error.
Test exact scenario from training to identify the source.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from src.models import LightGCNRecommender  
from src.losses import bpr_loss, l2_regularization
from src import config

print("🔍 DEBUG: 'int' object has no attribute 'dtype' error")
print("=" * 60)

try:
    # Simulate exact training scenario
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    num_users, num_items = 100, 50
    model = LightGCNRecommender(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=16,
        num_layers=2
    ).to(device)
    
    # Create test data
    edge_index = torch.tensor([[0, 1, 2], [50, 60, 70]], dtype=torch.long, device=device)
    users = torch.tensor([0, 1, 2], device=device)
    pos_items = torch.tensor([10, 15, 20], device=device)
    neg_items = torch.tensor([25, 30, 35], device=device)
    
    print("✅ Test data created")
    
    # Forward pass
    print("🔄 Testing forward pass...")
    pos_scores, neg_scores, embeddings = model.bpr_forward(
        edge_index, users, pos_items, neg_items
    )
    
    print("✅ Forward pass completed")
    print(f"   Embeddings type: {type(embeddings)}")
    
    # Unpack embeddings
    print("🔄 Unpacking embeddings...")
    u_emb, pos_emb, neg_emb = embeddings
    
    print(f"✅ Unpacked embeddings:")
    print(f"   u_emb: type={type(u_emb)}, shape={u_emb.shape}, dtype={u_emb.dtype}")
    print(f"   pos_emb: type={type(pos_emb)}, shape={pos_emb.shape}, dtype={pos_emb.dtype}")
    print(f"   neg_emb: type={type(neg_emb)}, shape={neg_emb.shape}, dtype={neg_emb.dtype}")
    
    # Test l2_regularization with different scenarios
    print("\n🧪 Testing L2 regularization scenarios...")
    
    # Scenario 1: Only tensors (should work)
    print("1️⃣ Testing with tensor embeddings only...")
    try:
        reg1 = l2_regularization(u_emb, pos_emb, neg_emb)
        print(f"   ✅ Tensor-only: {reg1.item():.4f}")
    except Exception as e:
        print(f"   ❌ Tensor-only failed: {e}")
    
    # Scenario 2: With batch_size (this was in original call)
    print("2️⃣ Testing with batch_size parameter...")
    try:
        batch_size = users.size(0)  # This returns Python int
        print(f"   batch_size = {batch_size}, type = {type(batch_size)}")
        
        reg2 = l2_regularization(u_emb, pos_emb, neg_emb, batch_size=batch_size)
        print(f"   ✅ With batch_size: {reg2.item():.4f}")
    except Exception as e:
        print(f"   ❌ With batch_size failed: {e}")
    
    # Scenario 3: Test with potential integer input
    print("3️⃣ Testing with integer input (should fail gracefully)...")
    try:
        reg3 = l2_regularization(u_emb, pos_emb, 42)  # Pass int instead of tensor
        print(f"   ❌ Should have failed but didn't: {reg3}")
    except Exception as e:
        print(f"   ✅ Correctly caught error: {e}")
    
    # Scenario 4: Test exact training call
    print("4️⃣ Testing exact training call...")
    try:
        # This is the exact call from ile_training.py line 177
        reg4 = l2_regularization(u_emb, pos_emb, neg_emb, users.size(0))
        print(f"   ✅ Exact training call: {reg4.item():.4f}")
    except Exception as e:
        print(f"   ❌ Exact training call failed: {e}")
    
    # Scenario 5: Test corrected training call
    print("5️⃣ Testing corrected training call...")
    try:
        # This is the corrected call
        reg5 = l2_regularization(u_emb, pos_emb, neg_emb, batch_size=users.size(0))
        print(f"   ✅ Corrected training call: {reg5.item():.4f}")
    except Exception as e:
        print(f"   ❌ Corrected training call failed: {e}")
    
except Exception as e:
    print(f"\n❌ DEBUG FAILED: {e}")
    import traceback
    traceback.print_exc()
    
    print(f"\n🎉 DEBUG COMPLETED!")
    print(f"✅ All tensor operations working correctly")