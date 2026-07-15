#!/usr/bin/env python3
"""
ILE Loss Functions - Item Loss Equalization Implementation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import config


def compute_item_popularity_groups(item_degrees: torch.Tensor, 
                                 tail_percentile: float = 50.0,
                                 head_percentile: float = 80.0) -> torch.Tensor:
    """
    Compute item popularity groups based on interaction degrees.
    
    Args:
        item_degrees: Item interaction counts [num_items]
        tail_percentile: Percentile cutoff for tail items (default 50%)
        head_percentile: Percentile cutoff for head items (default 80%)
        
    Returns:
        Item group labels [num_items] where 0=tail, 1=middle, 2=head
    """
    
    # Compute percentile thresholds
    p_tail = np.percentile(item_degrees.cpu().numpy(), tail_percentile)
    p_head = np.percentile(item_degrees.cpu().numpy(), head_percentile)
    
    # Assign groups
    groups = torch.zeros_like(item_degrees, dtype=torch.long)
    groups[item_degrees >= p_head] = config.GROUP_HEAD      # Head (top 20%)
    groups[(item_degrees >= p_tail) & (item_degrees < p_head)] = config.GROUP_MIDDLE  # Middle (30%)
    groups[item_degrees < p_tail] = config.GROUP_TAIL       # Tail (bottom 50%)
    
    return groups


def ile_loss(pos_scores: torch.Tensor, 
            neg_scores: torch.Tensor,
            pos_items: torch.Tensor,
            item_groups: torch.Tensor,
            device: torch.device) -> torch.Tensor:
    """
    Item Loss Equalization (ILE) loss.
    
    Computes the difference between head and tail item losses to promote fairness.
    
    Args:
        pos_scores: Positive item scores [batch_size]
        neg_scores: Negative item scores [batch_size] 
        pos_items: Positive item indices [batch_size]
        item_groups: Item group assignments [num_items]
        device: Torch device
        
    Returns:
        ILE loss scalar
    """
    
    if len(pos_items) == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    
    # CRITICAL FIX: Ensure all tensors are on the same device
    pos_scores = pos_scores.to(device)
    neg_scores = neg_scores.to(device)
    pos_items = pos_items.to(device)
    item_groups = item_groups.to(device)
    
    # CRITICAL FIX: Add numerical stability for sigmoid
    score_diff = pos_scores - neg_scores
    score_diff = torch.clamp(score_diff, min=-50, max=50)  # Prevent overflow
    
    # Get BPR losses for each sample (no reduction) with numerical stability
    sigmoid_scores = torch.sigmoid(score_diff)
    sigmoid_scores = torch.clamp(sigmoid_scores, min=1e-8, max=1-1e-8)  # Prevent log(0)
    bpr_losses = -torch.log(sigmoid_scores)
    
    # Get item groups for positive items
    # CRITICAL FIX: Enhanced bounds checking 
    valid_items_mask = (pos_items >= 0) & (pos_items < len(item_groups))
    if not valid_items_mask.all():
        invalid_count = (~valid_items_mask).sum().item()
        print(f"⚠️  Warning: {invalid_count} invalid item indices detected. Min: {pos_items.min()}, Max: {pos_items.max()}, Items range: [0, {len(item_groups)-1}]")
        
        # Filter invalid indices
        pos_items = pos_items[valid_items_mask]
        bpr_losses = bpr_losses[valid_items_mask]
        
        if len(pos_items) == 0:
            print("⚠️  Warning: No valid items after filtering, returning zero ILE loss")
            return torch.tensor(0.0, device=device, dtype=torch.float32)
    
    pos_item_groups = item_groups[pos_items]
    
    # Separate losses by item groups
    tail_mask = (pos_item_groups == config.GROUP_TAIL)
    head_mask = (pos_item_groups == config.GROUP_HEAD)
    
    # ILE cần CẢ hai nhóm head và tail trong batch mới cân bằng được
    tail_count = tail_mask.sum().item()
    head_count = head_mask.sum().item()
    if tail_count == 0 or head_count == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    tail_loss = bpr_losses[tail_mask].mean()
    head_loss = bpr_losses[head_mask].mean()

    # Item Loss Equalization: phạt độ CHÊNH LỆCH loss giữa hai nhóm (bình phương, luôn >= 0).
    # Minimize -> kéo tail_loss (thường cao hơn) xuống gần head_loss => cải thiện nhóm tail,
    # giảm popularity bias, chấp nhận trade-off nhẹ ở head.
    # (Bản cũ dùng head_loss - tail_loss: lật dấu + KHÔNG chặn dưới => càng train càng đẩy
    #  về item phổ biến, đúng như kết quả hỏng đã đo trước đây.)
    ile_penalty = (head_loss - tail_loss) ** 2
    
    # CRITICAL FIX: Ensure output is finite
    if not torch.isfinite(ile_penalty):
        print(f"⚠️  Warning: Non-finite ILE loss detected, returning zero")
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    
    return ile_penalty


def contrastive_loss(user_embeds: torch.Tensor,
                    pos_item_embeds: torch.Tensor, 
                    neg_item_embeds: torch.Tensor,
                    temperature: float = config.TAU) -> torch.Tensor:
    """
    InfoNCE contrastive loss for graph augmentation.
    
    Args:
        user_embeds: User embeddings [batch_size, embed_dim]
        pos_item_embeds: Positive item embeddings [batch_size, embed_dim]
        neg_item_embeds: Negative item embeddings [batch_size, embed_dim]
        temperature: Temperature parameter
        
    Returns:
        Contrastive loss scalar
    """
    
    # CRITICAL FIX: Ensure all tensors have the same device
    device = user_embeds.device
    pos_item_embeds = pos_item_embeds.to(device)
    neg_item_embeds = neg_item_embeds.to(device)
    
    # CRITICAL FIX: Add input validation
    if user_embeds.size(0) == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    
    if user_embeds.size(0) != pos_item_embeds.size(0) or user_embeds.size(0) != neg_item_embeds.size(0):
        raise ValueError(f"Batch size mismatch: users={user_embeds.size(0)}, pos_items={pos_item_embeds.size(0)}, neg_items={neg_item_embeds.size(0)}")
    
    # Normalize embeddings with numerical stability
    user_embeds = F.normalize(user_embeds, dim=1, eps=1e-8)
    pos_item_embeds = F.normalize(pos_item_embeds, dim=1, eps=1e-8)
    neg_item_embeds = F.normalize(neg_item_embeds, dim=1, eps=1e-8)
    
    # CRITICAL FIX: Ensure temperature is valid
    if temperature <= 0:
        print(f"⚠️  Warning: Invalid temperature {temperature}, using default")
        temperature = config.TAU
    
    # Positive similarities 
    pos_sim = torch.sum(user_embeds * pos_item_embeds, dim=1) / temperature
    
    # Negative similarities
    neg_sim = torch.sum(user_embeds * neg_item_embeds, dim=1) / temperature
    
    # CRITICAL FIX: Add numerical stability for logits
    pos_sim = torch.clamp(pos_sim, min=-50, max=50)
    neg_sim = torch.clamp(neg_sim, min=-50, max=50)
    
    # InfoNCE loss
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
    
    loss = F.cross_entropy(logits, labels)
    
    # CRITICAL FIX: Ensure output is finite
    if not torch.isfinite(loss):
        print(f"⚠️  Warning: Non-finite contrastive loss detected, returning zero")
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    
    return loss


def compute_degree_aware_dropout_probs(item_degrees: torch.Tensor,
                                     p_min: float = config.DROPOUT_P_MIN,
                                     p_max: float = config.DROPOUT_P_MAX) -> torch.Tensor:
    """
    Compute degree-aware dropout probabilities.
    
    Popular items (high degree) get higher dropout probability.
    
    Args:
        item_degrees: Item interaction degrees [num_items]
        p_min: Minimum dropout probability
        p_max: Maximum dropout probability
        
    Returns:
        Dropout probabilities [num_items]
    """
    
    # CRITICAL FIX: Input validation
    if len(item_degrees) == 0:
        return torch.tensor([], dtype=torch.float, device=item_degrees.device)
    
    # CRITICAL FIX: Ensure item_degrees is float tensor and handle device
    device = item_degrees.device
    item_degrees = item_degrees.float().to(device)
    max_degree = item_degrees.max()
    
    # CRITICAL FIX: Validate probability range
    if not (0 <= p_min <= p_max <= 1):
        raise ValueError(f"Invalid probability range: p_min={p_min}, p_max={p_max}")
    
    if max_degree == 0:
        return torch.full_like(item_degrees, p_min, dtype=torch.float, device=device)
    
    # CRITICAL FIX: Add numerical stability for log operations
    log_degrees = torch.log(1.0 + item_degrees.clamp(min=0))
    log_max = torch.log(1.0 + max_degree.clamp(min=0))
    
    # Handle edge case where log_max is 0
    if log_max == 0:
        return torch.full_like(item_degrees, p_min, dtype=torch.float, device=device)
    
    # Logarithmic scaling: p = p_min + (p_max - p_min) * log(1 + deg) / log(1 + deg_max)
    scaling_factor = log_degrees / log_max
    dropout_probs = p_min + (p_max - p_min) * scaling_factor
    
    # CRITICAL FIX: Ensure output is finite and in valid range
    dropout_probs = torch.clamp(dropout_probs, p_min, p_max)
    
    if not torch.isfinite(dropout_probs).all():
        print(f"⚠️  Warning: Non-finite dropout probabilities detected, using uniform probabilities")
        return torch.full_like(item_degrees, (p_min + p_max) / 2, dtype=torch.float, device=device)
    
    return dropout_probs