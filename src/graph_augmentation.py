#!/usr/bin/env python3
"""
Graph Augmentation Module - Degree-aware Edge Dropout
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import config
from ile_losses import compute_degree_aware_dropout_probs


def apply_degree_aware_dropout(edge_index: torch.Tensor,
                              item_degrees: torch.Tensor,
                              num_users: int,
                              dropout_type: str = 'degree_aware',
                              p_uniform: float = config.UNIFORM_DROPOUT_P,
                              p_min: float = config.DROPOUT_P_MIN,
                              p_max: float = config.DROPOUT_P_MAX,
                              training: bool = True) -> torch.Tensor:
    """
    Apply degree-aware or uniform edge dropout to bipartite graph.
    
    Args:
        edge_index: Bipartite graph edges [2, num_edges]
        item_degrees: Item interaction degrees [num_items]
        num_users: Number of users
        dropout_type: 'uniform' or 'degree_aware'
        p_uniform: Uniform dropout probability
        p_min: Min dropout probability for degree-aware
        p_max: Max dropout probability for degree-aware
        training: Whether in training mode
        
    Returns:
        Augmented edge_index with dropped edges
    """
    
    if not training:
        return edge_index
    
    # CRITICAL FIX: Input validation
    if edge_index.size(1) == 0:
        return edge_index
    
    device = edge_index.device
    
    # CRITICAL FIX: Ensure item_degrees is on correct device
    item_degrees = item_degrees.to(device)
    
    if dropout_type == 'uniform':
        # CRITICAL FIX: Validate probability
        if not (0 <= p_uniform <= 1):
            raise ValueError(f"Invalid uniform dropout probability: {p_uniform}")
            
        # Uniform dropout
        keep_prob = 1.0 - p_uniform
        keep_mask = torch.rand(edge_index.size(1), device=device) < keep_prob
        augmented_edge_index = edge_index[:, keep_mask]
        
    elif dropout_type == 'degree_aware':
        # Degree-aware dropout
        dropout_probs = compute_degree_aware_dropout_probs(
            item_degrees, p_min, p_max
        ).to(device)
        
        # CRITICAL FIX: Proper bipartite graph edge processing
        # In bipartite graph: edges are bidirectional
        # [user_nodes, item_nodes] and [item_nodes, user_nodes]
        # We need to identify edges that involve items and apply dropout based on item degrees
        
        # Get all edges
        sources = edge_index[0]  
        targets = edge_index[1]
        
        # Identify user->item edges (target >= num_users means item node)
        user_to_item_mask = targets >= num_users
        user_to_item_edges = edge_index[:, user_to_item_mask]
        
        # Identify item->user edges (source >= num_users means item node)
        item_to_user_mask = sources >= num_users
        item_to_user_edges = edge_index[:, item_to_user_mask]
        
        # Process user->item edges
        kept_user_to_item = user_to_item_edges
        if user_to_item_edges.size(1) > 0:
            item_indices_ui = user_to_item_edges[1] - num_users  # target nodes are items
            
            # Bounds checking
            valid_mask_ui = (item_indices_ui >= 0) & (item_indices_ui < len(item_degrees))
            if not valid_mask_ui.all():
                invalid_count = (~valid_mask_ui).sum().item()
                print(f"⚠️  Warning: {invalid_count} invalid user->item edges. Filtering...")
                user_to_item_edges = user_to_item_edges[:, valid_mask_ui]
                item_indices_ui = item_indices_ui[valid_mask_ui]
            
            if len(item_indices_ui) > 0:
                edge_dropout_probs_ui = dropout_probs[item_indices_ui]
                keep_probs_ui = 1.0 - edge_dropout_probs_ui
                keep_mask_ui = torch.rand(len(item_indices_ui), device=device) < keep_probs_ui
                kept_user_to_item = user_to_item_edges[:, keep_mask_ui]
        
        # Process item->user edges 
        kept_item_to_user = item_to_user_edges
        if item_to_user_edges.size(1) > 0:
            item_indices_iu = item_to_user_edges[0] - num_users  # source nodes are items
            
            # Bounds checking
            valid_mask_iu = (item_indices_iu >= 0) & (item_indices_iu < len(item_degrees))
            if not valid_mask_iu.all():
                invalid_count = (~valid_mask_iu).sum().item()
                print(f"⚠️  Warning: {invalid_count} invalid item->user edges. Filtering...")
                item_to_user_edges = item_to_user_edges[:, valid_mask_iu]
                item_indices_iu = item_indices_iu[valid_mask_iu]
            
            if len(item_indices_iu) > 0:
                edge_dropout_probs_iu = dropout_probs[item_indices_iu]
                keep_probs_iu = 1.0 - edge_dropout_probs_iu
                keep_mask_iu = torch.rand(len(item_indices_iu), device=device) < keep_probs_iu
                kept_item_to_user = item_to_user_edges[:, keep_mask_iu]
        
        # Combine kept edges
        edge_list = []
        if kept_user_to_item.size(1) > 0:
            edge_list.append(kept_user_to_item)
        if kept_item_to_user.size(1) > 0:
            edge_list.append(kept_item_to_user)
            
        if len(edge_list) > 0:
            augmented_edge_index = torch.cat(edge_list, dim=1)
        else:
            # If all edges dropped, return empty tensor with correct shape
            augmented_edge_index = torch.empty((2, 0), dtype=edge_index.dtype, device=device)
        
    else:
        raise ValueError(f"Unknown dropout_type: {dropout_type}")
    
    return augmented_edge_index


def create_augmented_views(edge_index: torch.Tensor,
                          item_degrees: torch.Tensor, 
                          num_users: int,
                          dropout_type: str = 'degree_aware',
                          num_views: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create multiple augmented views of the graph for contrastive learning.
    
    Args:
        edge_index: Original bipartite graph [2, num_edges]
        item_degrees: Item interaction degrees [num_items]
        num_users: Number of users
        dropout_type: Type of dropout to apply
        num_views: Number of augmented views to create
        
    Returns:
        Tuple of augmented edge indices
    """
    
    # CRITICAL FIX: Input validation
    if edge_index.size(1) == 0:
        empty_tensor = torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        return empty_tensor, empty_tensor
    
    if num_views != 2:
        raise ValueError(f"Only 2 views supported, got {num_views}")
    
    view1 = apply_degree_aware_dropout(
        edge_index, item_degrees, num_users, dropout_type, training=True
    )
    
    view2 = apply_degree_aware_dropout(
        edge_index, item_degrees, num_users, dropout_type, training=True
    )
    
    return view1, view2


class GraphAugmentation:
    """Graph augmentation handler with multiple strategies."""
    
    def __init__(self, 
                 num_users: int,
                 num_items: int,
                 dropout_type: str = 'degree_aware',
                 p_uniform: float = config.UNIFORM_DROPOUT_P,
                 p_min: float = config.DROPOUT_P_MIN,
                 p_max: float = config.DROPOUT_P_MAX):
        
        self.num_users = num_users
        self.num_items = num_items
        self.dropout_type = dropout_type
        self.p_uniform = p_uniform
        self.p_min = p_min
        self.p_max = p_max
        
    def augment(self, 
                edge_index: torch.Tensor,
                item_degrees: torch.Tensor,
                training: bool = True) -> torch.Tensor:
        """Apply augmentation to edge index."""
        
        return apply_degree_aware_dropout(
            edge_index=edge_index,
            item_degrees=item_degrees,
            num_users=self.num_users,
            dropout_type=self.dropout_type,
            p_uniform=self.p_uniform,
            p_min=self.p_min,
            p_max=self.p_max,
            training=training
        )
        
    def create_views(self,
                    edge_index: torch.Tensor,
                    item_degrees: torch.Tensor,
                    num_views: int = 2) -> Tuple[torch.Tensor, ...]:
        """Create multiple augmented views."""
        
        views = []
        for _ in range(num_views):
            view = self.augment(edge_index, item_degrees, training=True)
            views.append(view)
            
        return tuple(views)


def compute_graph_statistics(edge_index: torch.Tensor, 
                            num_users: int, 
                            num_items: int) -> dict:
    """Compute graph statistics for analysis."""
    
    # CRITICAL FIX: Input validation
    if edge_index.size(1) == 0:
        return {
            'num_edges': 0,
            'num_users': num_users,
            'num_items': num_items,
            'density': 0.0,
            'avg_user_degree': 0.0,
            'avg_item_degree': 0.0,
            'max_user_degree': 0,
            'max_item_degree': 0
        }
    
    stats = {}
    
    # Basic stats
    stats['num_edges'] = edge_index.size(1)
    stats['num_users'] = num_users
    stats['num_items'] = num_items
    
    # CRITICAL FIX: Protect against division by zero
    max_edges = num_users * num_items
    stats['density'] = stats['num_edges'] / max_edges if max_edges > 0 else 0.0
    
    # CRITICAL FIX: Proper degree calculation with bounds checking
    num_nodes = num_users + num_items
    
    # User degrees (sources < num_users)
    user_mask = edge_index[0] < num_users
    if user_mask.any():
        user_sources = edge_index[0][user_mask]
        user_degrees = torch.bincount(user_sources, minlength=num_users)
    else:
        user_degrees = torch.zeros(num_users, dtype=torch.long)
    
    # Item degrees: need to check both directions
    # user->item edges: targets >= num_users  
    # item->user edges: sources >= num_users
    item_interactions = torch.zeros(num_items, dtype=torch.long, device=edge_index.device)
    
    # From user->item edges (targets are item nodes)
    ui_mask = edge_index[1] >= num_users
    if ui_mask.any():
        item_targets = edge_index[1][ui_mask] - num_users
        valid_targets = (item_targets >= 0) & (item_targets < num_items)
        if valid_targets.any():
            valid_item_targets = item_targets[valid_targets] 
            item_degrees_ui = torch.bincount(valid_item_targets, minlength=num_items)
            item_interactions += item_degrees_ui
    
    # From item->user edges (sources are item nodes) 
    iu_mask = edge_index[0] >= num_users
    if iu_mask.any():
        item_sources = edge_index[0][iu_mask] - num_users
        valid_sources = (item_sources >= 0) & (item_sources < num_items)
        if valid_sources.any():
            valid_item_sources = item_sources[valid_sources]
            item_degrees_iu = torch.bincount(valid_item_sources, minlength=num_items)
            item_interactions += item_degrees_iu
    
    # Since edges are bidirectional, divide by 2 to get actual item degrees
    item_degrees = item_interactions // 2
    
    # CRITICAL FIX: Safe computation of statistics
    stats['avg_user_degree'] = user_degrees.float().mean().item() if len(user_degrees) > 0 else 0.0
    stats['avg_item_degree'] = item_degrees.float().mean().item() if len(item_degrees) > 0 else 0.0
    stats['max_user_degree'] = user_degrees.max().item() if len(user_degrees) > 0 else 0
    stats['max_item_degree'] = item_degrees.max().item() if len(item_degrees) > 0 else 0
    
    return stats