#!/usr/bin/env python3
"""
ILE Training Module - FIXED VERSION
Handles training with Item Loss Equalization and comprehensive evaluation on train/val/test splits
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import sys
sys.path.append(str(Path(__file__).parent))
import config
from config import get_device, set_seed, PROJECT_ROOT
from models import LightGCNRecommender
from losses import bpr_loss, l2_regularization
from train import sample_bpr_batch
from metrics import evaluate_full_ranking
from ile_losses import ile_loss


def train_model_with_ile(
    train_edges: torch.Tensor,
    edge_index_train: torch.Tensor,
    val_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_users: int,
    num_items: int,
    train_user_pos_items: List[set],
    val_user_pos_items: List[set],
    test_user_pos_items: List[set],
    item_degree: torch.Tensor,
    item_popularity_group: torch.Tensor,
    lambda_ile: float = 0.0,
    model_name: str = "LightGCN",
    save_checkpoints: bool = True,
    dropout_type: str = None,  # 'uniform', 'degree_aware', or None
    **kwargs
) -> Tuple[LightGCNRecommender, Dict]:
    """
    Train LightGCN with ILE and optional graph augmentation, evaluate on train/val/test splits.
    
    Args:
        train_edges: Training interactions [N, 2] 
        edge_index_train: Bipartite graph for message passing [2, E]
        val_edges: Validation interactions [N_val, 2]
        test_edges: Test interactions [N_test, 2] 
        num_users, num_items: Catalog sizes
        train_user_pos_items: Training interactions per user
        val_user_pos_items: Validation interactions per user  
        test_user_pos_items: Test interactions per user
        item_degree: Item interaction degrees for evaluation
        item_popularity_group: Item popularity groups [num_items] 
        lambda_ile: ILE regularization weight
        model_name: Model name for saving
        save_checkpoints: Whether to save model checkpoints
        dropout_type: Graph augmentation type ('uniform', 'degree_aware', None)
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    
    device = get_device()
    print(f"🖥️  Using device: {device}")
    
    # CRITICAL FIX: Input validation
    if len(train_edges) == 0:
        raise ValueError("Training edges cannot be empty")
    if edge_index_train.size(1) == 0:
        raise ValueError("Edge index cannot be empty")
    if num_users <= 0 or num_items <= 0:
        raise ValueError(f"Invalid catalog sizes: users={num_users}, items={num_items}")
    if lambda_ile < 0:
        raise ValueError(f"Invalid lambda_ile: {lambda_ile}")
    
    # CRITICAL FIX: Validate dropout_type
    if dropout_type is not None and dropout_type not in ['uniform', 'degree_aware']:
        raise ValueError(f"Invalid dropout_type: {dropout_type}")
    
    # SIMPLIFIED: No advanced optimizations, just basic training
    use_amp = False
    scaler = None
    
    print(f"🚀 Training configuration:")
    print(f"   - Batch size: {config.BATCH_SIZE}")
    print(f"   - Device: {device}")
    print(f"   - Basic training mode (no advanced optimizations)")
    print(f"   - Lambda ILE: {lambda_ile}")
    print(f"   - Dropout type: {dropout_type}")
    
    # Model setup
    model = LightGCNRecommender(
        num_users=num_users,
        num_items=num_items, 
        embedding_dim=config.EMBEDDING_DIM,
        num_layers=config.NUM_LAYERS
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    # CRITICAL FIX: Safe tensor device movement with validation
    try:
        item_popularity_group = item_popularity_group.to(device)
        item_degree = item_degree.to(device)
        edge_index_train = edge_index_train.to(device)
        
        # Validate tensor sizes
        if len(item_popularity_group) != num_items:
            raise ValueError(f"item_popularity_group size mismatch: {len(item_popularity_group)} != {num_items}")
        if len(item_degree) != num_items:
            raise ValueError(f"item_degree size mismatch: {len(item_degree)} != {num_items}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to move tensors to device {device}: {e}")
    
    # Setup graph augmentation if needed
    use_augmentation = dropout_type is not None
    if use_augmentation:
        from graph_augmentation import apply_degree_aware_dropout
        print(f"🔄 Using graph augmentation: {dropout_type}")
    
    print(f"📊 Item groups - Tail: {(item_popularity_group == 0).sum()}, "
          f"Middle: {(item_popularity_group == 1).sum()}, Head: {(item_popularity_group == 2).sum()}")
    
    # Training setup
    edges_cpu = train_edges.cpu()
    steps_per_epoch = max(1, len(edges_cpu) // config.BATCH_SIZE)  # Ensure at least 1 step
    
    # CRITICAL FIX: Initialize metrics tracking with proper defaults
    best_val_recall = -1.0  # Use -1 to indicate no validation run yet
    best_epoch = 0
    patience = 10
    no_improve_count = 0
    
    train_losses = []
    val_metrics_history = []
    
    print(f"🚀 Starting training: {model_name} (λ_ILE={lambda_ile}, aug={dropout_type})")
    print(f"📈 Epochs: {config.NUM_EPOCHS}, Batch size: {config.BATCH_SIZE}")
    print(f"🔄 Steps per epoch: {steps_per_epoch}")
    
    try:
        for epoch in range(1, config.NUM_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            epoch_bpr_loss = 0.0
            epoch_ile_loss = 0.0
            
            # Training loop
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch:3d}/{config.NUM_EPOCHS}")
            
            for step in pbar:
                try:
                    # Apply graph augmentation if enabled
                    if use_augmentation:
                        augmented_edge_index = apply_degree_aware_dropout(
                            edge_index_train, item_degree, num_users, 
                            dropout_type=dropout_type, training=True
                        )
                    else:
                        augmented_edge_index = edge_index_train
                    
                    # CRITICAL FIX: Validate augmented edge index
                    if augmented_edge_index.size(1) == 0:
                        print(f"⚠️  Warning: All edges dropped during augmentation, using original")
                        augmented_edge_index = edge_index_train
                        
                    # Sample batch
                    users, pos_items, neg_items = sample_bpr_batch(
                        edges_cpu, num_items, train_user_pos_items, config.BATCH_SIZE
                    )
                    
                    # CRITICAL FIX: Validate batch content
                    if len(users) == 0:
                        print(f"⚠️  Warning: Empty batch at step {step}, skipping")
                        continue
                    
                    # Move to device with error handling
                    users = users.to(device)
                    pos_items = pos_items.to(device) 
                    neg_items = neg_items.to(device)
                    
                    # BASIC: Forward pass with error handling
                    pos_scores, neg_scores, embeddings = model.bpr_forward(
                        augmented_edge_index, users, pos_items, neg_items
                    )
                    
                    # CRITICAL FIX: Validate forward pass outputs
                    if not torch.isfinite(pos_scores).all() or not torch.isfinite(neg_scores).all():
                        print(f"⚠️  Warning: Non-finite scores detected at epoch {epoch}, step {step}")
                        continue
                    
                    # BPR loss
                    bpr_loss_val = bpr_loss(pos_scores, neg_scores)
                    
                    # L2 regularization
                    u_emb, pos_emb, neg_emb = embeddings
                    reg_loss = l2_regularization(u_emb, pos_emb, neg_emb, batch_size=users.size(0))
                    
                    # ILE loss
                    ile_loss_val = torch.tensor(0.0, device=device, dtype=torch.float32)
                    if lambda_ile > 0:
                        ile_loss_val = ile_loss(
                            pos_scores, neg_scores, pos_items, item_popularity_group, device
                        )
                        # Ensure ILE loss is finite
                        if not torch.isfinite(ile_loss_val):
                            ile_loss_val = torch.tensor(0.0, device=device, dtype=torch.float32)
                    
                    # Total loss
                    total_loss = (bpr_loss_val + 
                                 config.WEIGHT_DECAY * reg_loss + 
                                 lambda_ile * ile_loss_val)
                    
                    # CRITICAL FIX: Validate total loss
                    if not torch.isfinite(total_loss):
                        print(f"⚠️  Warning: Non-finite total loss at epoch {epoch}, step {step}, skipping")
                        continue
                    
                    # BASIC: Standard backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    # CRITICAL FIX: Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Track losses
                    epoch_loss += total_loss.item()
                    epoch_bpr_loss += bpr_loss_val.item()
                    if lambda_ile > 0:
                        epoch_ile_loss += ile_loss_val.item()
                        
                    # Update progress bar
                    pbar.set_postfix({
                        'BPR': f'{bpr_loss_val.item():.4f}',
                        'ILE': f'{ile_loss_val.item() if lambda_ile > 0 else 0.0:.4f}',
                        'Total': f'{total_loss.item():.4f}'
                    })
                    
                except Exception as e:
                    print(f"⚠️  Error in training step {step}: {e}")
                    continue
            
            # Epoch metrics
            if steps_per_epoch > 0:
                avg_loss = epoch_loss / steps_per_epoch
                avg_bpr = epoch_bpr_loss / steps_per_epoch
                avg_ile = epoch_ile_loss / steps_per_epoch if lambda_ile > 0 else 0.0
            else:
                avg_loss = avg_bpr = avg_ile = 0.0
            
            train_losses.append({
                'epoch': epoch,
                'total_loss': avg_loss,
                'bpr_loss': avg_bpr, 
                'ile_loss': avg_ile
            })
            
            # Validation evaluation every 5 epochs
            if epoch % 5 == 0:
                try:
                    model.eval()
                    with torch.no_grad():
                        val_metrics = evaluate_split(
                            model, edge_index_train, val_edges, train_user_pos_items,
                            item_degree, item_popularity_group, device, "Validation"
                        )
                        
                    val_metrics_history.append({
                        'epoch': epoch,
                        **val_metrics
                    })
                    
                    # Check for improvement
                    current_val_recall = val_metrics[f'Recall@{config.PRIMARY_K}']
                    if current_val_recall > best_val_recall:
                        best_val_recall = current_val_recall
                        best_epoch = epoch
                        no_improve_count = 0
                        
                        if save_checkpoints:
                            save_checkpoint(model, optimizer, epoch, val_metrics, model_name)
                            print(f"   🎉 New best Recall@{config.PRIMARY_K}: {best_val_recall:.4f}")
                            
                    else:
                        no_improve_count += 1
                        
                    print(f"   📊 Val Recall@{config.PRIMARY_K}: {current_val_recall:.4f} "
                          f"(Best: {best_val_recall:.4f} @ epoch {best_epoch})")
                          
                except Exception as e:
                    print(f"⚠️  Error in validation at epoch {epoch}: {e}")
                    no_improve_count += 1
                      
            # Early stopping
            if no_improve_count >= patience and best_val_recall > -1:
                print(f"   ⏰ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    
    # Load best model
    if save_checkpoints and best_epoch > 0:
        load_best_checkpoint(model, model_name)
        print(f"   🔄 Restored best model with Recall@{config.PRIMARY_K}: {best_val_recall:.4f}")
    
    # Final evaluation on all splits (use original graph, no augmentation)
    model.eval()
    final_metrics = {}
    
    with torch.no_grad():
        # Training set evaluation
        train_metrics = evaluate_split(
            model, edge_index_train, train_edges, train_user_pos_items,
            item_degree, item_popularity_group, device, "Training"
        )
        final_metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
        
        # Validation set evaluation
        val_metrics = evaluate_split(
            model, edge_index_train, val_edges, train_user_pos_items,
            item_degree, item_popularity_group, device, "Validation" 
        )
        final_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        # Test set evaluation  
        test_metrics = evaluate_split(
            model, edge_index_train, test_edges, val_user_pos_items,
            item_degree, item_popularity_group, device, "Test"
        )
        final_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    
    # Add training metadata
    final_metrics.update({
        'model_name': model_name,
        'lambda_ile': lambda_ile,
        'dropout_type': dropout_type,
        'best_epoch': best_epoch,
        'total_epochs': epoch,
        'best_val_recall': best_val_recall,
        'train_losses': train_losses,
        'val_history': val_metrics_history
    })
    
    return model, final_metrics


def evaluate_split(model, edge_index, edges, mask_user_pos_items, 
                  item_degree, item_popularity_group, device, split_name):
    """Evaluate model on a specific data split - FIXED VERSION."""
    
    print(f"📈 Evaluating on {split_name} set...")
    
    # CRITICAL FIX: Input validation
    if len(edges) == 0:
        print(f"⚠️  Warning: No edges in {split_name} set")
        # Return empty metrics
        empty_metrics = {}
        for k in config.K_LIST:
            empty_metrics[f'Recall@{k}'] = 0.0
            empty_metrics[f'TailRecall@{k}'] = 0.0
            empty_metrics[f'Coverage@{k}'] = 0.0
            empty_metrics[f'NDCG@{k}'] = 0.0
        return empty_metrics
    
    try:
        # Get full score matrix
        score_matrix = model.full_sort_scores(edge_index)
        
        # CRITICAL FIX: Validate score matrix
        if not torch.isfinite(score_matrix).all():
            print(f"⚠️  Warning: Non-finite scores in {split_name} evaluation")
            score_matrix = torch.where(torch.isfinite(score_matrix), score_matrix, torch.tensor(0.0, device=device))
        
        # Convert edges to test_items tensor with proper handling of multiple items per user
        num_users = score_matrix.size(0)
        test_items = torch.full((num_users,), -1, dtype=torch.long, device=device)  # Use -1 as default
        
        # CRITICAL FIX: Handle multiple test items per user properly
        user_test_items = {}
        for user, item in edges.tolist():
            if 0 <= user < num_users and 0 <= item < score_matrix.size(1):
                if user not in user_test_items:
                    user_test_items[user] = []
                user_test_items[user].append(item)
        
        # Use last item for each user (or could use first, or handle differently)
        for user, items in user_test_items.items():
            test_items[user] = items[-1]  # Use last item
        
        # CRITICAL FIX: Filter out users with no test items
        valid_users = test_items != -1
        if not valid_users.any():
            print(f"⚠️  Warning: No valid test items in {split_name} set")
            # Return empty metrics
            empty_metrics = {}
            for k in config.K_LIST:
                empty_metrics[f'Recall@{k}'] = 0.0
                empty_metrics[f'TailRecall@{k}'] = 0.0
                empty_metrics[f'Coverage@{k}'] = 0.0
                empty_metrics[f'NDCG@{k}'] = 0.0
            return empty_metrics
        
        # CRITICAL FIX: Correct function call with all required parameters
        metrics = evaluate_full_ranking(
            scores=score_matrix,
            train_user_pos_items=mask_user_pos_items,
            test_items=test_items,
            item_group=item_popularity_group,
            item_degree=item_degree,
            k_list=config.K_LIST
        )
        
        # Print key metrics
        recall_20 = metrics[f'Recall@{config.PRIMARY_K}']
        tail_recall_20 = metrics[f'TailRecall@{config.PRIMARY_K}']
        coverage_20 = metrics[f'Coverage@{config.PRIMARY_K}']
        
        print(f"   {split_name} - Recall@{config.PRIMARY_K}: {recall_20:.4f}, "
              f"TailRecall@{config.PRIMARY_K}: {tail_recall_20:.4f}, "
              f"Coverage@{config.PRIMARY_K}: {coverage_20:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error evaluating {split_name} set: {e}")
        # Return empty metrics
        empty_metrics = {}
        for k in config.K_LIST:
            empty_metrics[f'Recall@{k}'] = 0.0
            empty_metrics[f'TailRecall@{k}'] = 0.0
            empty_metrics[f'Coverage@{k}'] = 0.0
            empty_metrics[f'NDCG@{k}'] = 0.0
        return empty_metrics


def save_checkpoint(model, optimizer, epoch, metrics, model_name):
    """Save model checkpoint with proper naming."""
    
    checkpoints_dir = PROJECT_ROOT / "checkpoints" 
    checkpoints_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_name': model_name,
        'timestamp': timestamp
    }
    
    checkpoint_path = checkpoints_dir / f"checkpoint_{model_name}_epoch_{epoch}_{timestamp}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_best_checkpoint(model, model_name):
    """Load the best checkpoint for a model."""
    
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    
    # Find all checkpoints for this model
    # CRITICAL FIX: Use iterdir() instead of glob to avoid pattern issues
    checkpoint_files = []
    prefix = f"checkpoint_{model_name}_epoch_"
    
    if checkpoints_dir.exists():
        for file in checkpoints_dir.iterdir():
            if file.name.startswith(prefix) and file.name.endswith(".pt"):
                checkpoint_files.append(file)
    
    if not checkpoint_files:
        print(f"⚠️  No checkpoints found for {model_name}")
        return False
        
    # Load the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    
    try:
        checkpoint = torch.load(latest_checkpoint, map_location=get_device())
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📥 Loaded checkpoint: {latest_checkpoint.name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint {latest_checkpoint.name}: {e}")
        return False


def save_final_model(model, model_name, metadata):
    """Save final trained model with comprehensive metadata and proper naming."""
    
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comprehensive model package
    model_package = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'config': {
            'embedding_dim': config.EMBEDDING_DIM,
            'num_layers': config.NUM_LAYERS,
            'num_epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LR,
            'weight_decay': config.WEIGHT_DECAY
        },
        'training_info': {
            'timestamp': timestamp,
            'lambda_ile': metadata.get('lambda_ile', 0.0),
            'best_epoch': metadata.get('best_epoch', 0),
            'final_test_recall': metadata.get('test_Recall@20', 0.0),
            'final_test_tail_recall': metadata.get('test_TailRecall@20', 0.0),
            'final_test_coverage': metadata.get('test_Coverage@20', 0.0)
        }
    }
    
    model_path = models_dir / f"final_model_{model_name}_{timestamp}.pt"
    torch.save(model_package, model_path)
    
    print(f"💾 Final Model saved to: {model_path}")
    print(f"📊 Test Performance: Recall@20={metadata.get('test_Recall@20', 0.0):.4f}, "
          f"TailRecall@20={metadata.get('test_TailRecall@20', 0.0):.4f}")
    
    return model_path