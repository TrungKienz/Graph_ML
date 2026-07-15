#!/usr/bin/env python3
"""
Verify that the 'int' pow error is completely fixed.
Simulates the exact scenario that was causing the error.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from src.models import LightGCNRecommender
from src.losses import bpr_loss, l2_regularization
from src import config

print("🔍 VERIFYING POW ERROR FIX")
print("=" * 50)

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create the exact scenario from training
    print("\n1️⃣ Creating model and data...")
    num_users, num_items = 100, 50
    model = LightGCNRecommender(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=16,  # Small for testing
        num_layers=2
    ).to(device)
    
    # Create test edge index
    edge_index = torch.tensor([[0, 1, 2], [25, 30, 35]], dtype=torch.long, device=device)
    
    # Create test batch
    batch_users = torch.tensor([0, 1, 2], device=device)
    batch_pos = torch.tensor([25, 30, 35], device=device)
    batch_neg = torch.tensor([40, 45, 48], device=device)
    
    print("✅ Model and data created")
    
    print("\n2️⃣ Testing forward pass...")
    pos_scores, neg_scores, embeddings = model.bpr_forward(
        edge_index, batch_users, batch_pos, batch_neg
    )
    print("✅ Forward pass completed")
    
    print("\n3️⃣ Testing BPR loss...")
    bpr_loss_val = bpr_loss(pos_scores, neg_scores)
    print(f"✅ BPR loss: {bpr_loss_val.item():.4f}")
    
    print("\n4️⃣ Testing L2 regularization (this was the error source)...")
    u_emb, pos_emb, neg_emb = embeddings
    
    print(f"   Embedding dtypes: u={u_emb.dtype}, pos={pos_emb.dtype}, neg={neg_emb.dtype}")
    
    reg_loss = l2_regularization(u_emb, pos_emb, neg_emb, batch_users.size(0))
    print(f"✅ L2 regularization: {reg_loss.item():.4f}")
    
    print("\n5️⃣ Testing total loss and backward pass...")
    total_loss = bpr_loss_val + config.WEIGHT_DECAY * reg_loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"✅ Backward pass completed: {total_loss.item():.4f}")
    
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"✅ The 'int' object has no attribute 'pow' error is FIXED!")
    print(f"✅ Training should now work without errors.")
    
except Exception as e:
    print(f"\n❌ ERROR STILL EXISTS: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)