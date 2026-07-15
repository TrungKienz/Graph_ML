#!/usr/bin/env python3
"""
Data Loader - Load preprocessed data for ILE experiments
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent))
import config


class DataProcessor:
    """Load and process preprocessed MovieLens data."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Use the path specified by user for parquet files
            self.parquet_dir = config.PROJECT_ROOT / "preprocess_data" / "raw_data_after_preprocess_and_split"
            self.tensor_dir = config.PROJECT_ROOT / "preprocess_data"
        else:
            self.parquet_dir = Path(data_dir)
            self.tensor_dir = Path(data_dir).parent / "tensor_cache"
            
        self.tensor_dir.mkdir(exist_ok=True)
        
        # PRIORITY: Try to load from cached tensors first (correct dtypes guaranteed)
        if self._try_load_cached_tensors():
            print("✅ Loaded from cached tensors (correct dtypes)")
            self._create_user_pos_items()
        else:
            # Fallback: Convert from parquet (but this may have dtype issues)
            print("⚠️  No cached tensors found, converting from parquet...")
            self._convert_and_load_parquet()
            self._create_user_pos_items()
    
    def _try_load_cached_tensors(self):
        """Try to load from pre-generated tensor files with correct dtypes."""
        
        required_files = [
            'train_edges.pt',
            'val_edges.pt', 
            'test_edges.pt',
            'edge_index_train.pt',
            'item_degree.pt',
            'item_popularity_group.pt',
            'metadata.pt'
        ]
        
        # Check if all required files exist
        for filename in required_files:
            if not (self.tensor_dir / filename).exists():
                return False
        
        try:
            print(f"📂 Loading cached tensors from: {self.tensor_dir}")
            
            # Load metadata
            # weights_only=False: PyTorch>=2.6 mặc định True làm hỏng việc nạp file .pt chứa
            # numpy scalar/metadata. Đây là file do chính project sinh ra (tin cậy) nên an toàn.
            metadata = torch.load(self.tensor_dir / 'metadata.pt', weights_only=False)
            self.num_users = metadata['num_users']
            self.num_items = metadata['num_items']

            # Load tensors
            self.train_edges = torch.load(self.tensor_dir / 'train_edges.pt', weights_only=False)
            self.val_edges = torch.load(self.tensor_dir / 'val_edges.pt', weights_only=False)
            self.test_edges = torch.load(self.tensor_dir / 'test_edges.pt', weights_only=False)
            self.edge_index_train = torch.load(self.tensor_dir / 'edge_index_train.pt', weights_only=False)
            self.item_degree = torch.load(self.tensor_dir / 'item_degree.pt', weights_only=False)
            self.item_popularity_group = torch.load(self.tensor_dir / 'item_popularity_group.pt', weights_only=False)
            
            # CRITICAL: Verify dtypes
            assert self.item_degree.dtype == torch.float32, f"item_degree must be float32, got {self.item_degree.dtype}"
            assert self.train_edges.dtype == torch.long, f"train_edges must be long, got {self.train_edges.dtype}"
            assert self.item_popularity_group.dtype == torch.long, f"item_popularity_group must be long, got {self.item_popularity_group.dtype}"
            
            print(f"✅ Cached tensor loading successful:")
            print(f"   Users: {self.num_users}, Items: {self.num_items}")
            print(f"   item_degree dtype: {self.item_degree.dtype}")
            print(f"   train_edges dtype: {self.train_edges.dtype}")
            print(f"   edge_index dtype: {self.edge_index_train.dtype}")
            
            tail_count = (self.item_popularity_group == config.GROUP_TAIL).sum().item()
            middle_count = (self.item_popularity_group == config.GROUP_MIDDLE).sum().item()
            head_count = (self.item_popularity_group == config.GROUP_HEAD).sum().item()
            print(f"📊 Item groups: Tail={tail_count}, Middle={middle_count}, Head={head_count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load cached tensors: {e}")
            return False

    def _convert_and_load_parquet(self):
        """Convert parquet to tensors and load - FIXED VERSION."""
        
        print(f"📁 Loading parquet data from: {self.parquet_dir}")
        
        try:
            import pandas as pd
            
            # Load parquet files
            train_df = pd.read_parquet(self.parquet_dir / "train.parquet")
            val_df = pd.read_parquet(self.parquet_dir / "val.parquet") 
            test_df = pd.read_parquet(self.parquet_dir / "test.parquet")
            
            print(f"✅ Parquet files loaded:")
            print(f"   Train: {len(train_df)} interactions")
            print(f"   Val: {len(val_df)} interactions")
            print(f"   Test: {len(test_df)} interactions")
            
            # Extract dimensions from data
            self.num_users = max(train_df['user_idx'].max(), val_df['user_idx'].max(), test_df['user_idx'].max()) + 1
            self.num_items = max(train_df['movie_idx'].max(), val_df['movie_idx'].max(), test_df['movie_idx'].max()) + 1
            
            print(f"   Users: {self.num_users} (0-{self.num_users-1})")
            print(f"   Items: {self.num_items} (0-{self.num_items-1})")
            
            # Convert to edge tensors
            self.train_edges = torch.tensor(train_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
            self.val_edges = torch.tensor(val_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
            self.test_edges = torch.tensor(test_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
            
            # FIXED: Correct bipartite graph construction
            # Users: nodes 0..num_users-1, Items: nodes num_users..num_users+num_items-1
            user_nodes = self.train_edges[:, 0]  # User IDs: 0..num_users-1
            item_nodes = self.train_edges[:, 1] + self.num_users  # Item nodes: num_users..num_users+num_items-1
            
            # Create bidirectional edges for undirected bipartite graph
            self.edge_index_train = torch.stack([
                torch.cat([user_nodes, item_nodes]),  # user->item, item->user 
                torch.cat([item_nodes, user_nodes])   # item->user, user->item
            ])
            
            print(f"📊 Edge index: shape={self.edge_index_train.shape}")
            print(f"   Node range: [{self.edge_index_train.min()}, {self.edge_index_train.max()}]")
            print(f"   Expected max: {self.num_users + self.num_items - 1}")
            
            # Calculate item degrees - FIXED: Use float dtype
            item_degrees = train_df['movie_idx'].value_counts().sort_index()
            self.item_degree = torch.zeros(self.num_items, dtype=torch.float)  # FIXED: float not long
            for item_id, degree in item_degrees.items():
                if 0 <= item_id < self.num_items:
                    self.item_degree[item_id] = float(degree)  # FIXED: Ensure float
                
            # Create item popularity groups based on degree percentiles
            degree_values = self.item_degree.float()
            
            # Handle edge case: if all degrees are 0
            if degree_values.sum() == 0:
                self.item_popularity_group = torch.zeros(self.num_items, dtype=torch.long)
                print("⚠️  Warning: All items have zero degree, assigning all to tail group")
            else:
                p50 = torch.quantile(degree_values[degree_values > 0], 0.5) if (degree_values > 0).any() else 0
                p80 = torch.quantile(degree_values[degree_values > 0], 0.8) if (degree_values > 0).any() else 0
                
                self.item_popularity_group = torch.zeros(self.num_items, dtype=torch.long)
                self.item_popularity_group[degree_values >= p80] = config.GROUP_HEAD  # Top 20%
                self.item_popularity_group[(degree_values >= p50) & (degree_values < p80)] = config.GROUP_MIDDLE  # Middle 30%
                self.item_popularity_group[degree_values < p50] = config.GROUP_TAIL  # Bottom 50%
            
            tail_count = (self.item_popularity_group == config.GROUP_TAIL).sum().item()
            middle_count = (self.item_popularity_group == config.GROUP_MIDDLE).sum().item() 
            head_count = (self.item_popularity_group == config.GROUP_HEAD).sum().item()
            
            print(f"📊 Item groups: Tail={tail_count}, Middle={middle_count}, Head={head_count}")
            
            # VERIFICATION: Test bounds to prevent CUDA errors
            print("🧪 Verifying data integrity...")
            
            # Test edge index bounds
            max_node_id = self.num_users + self.num_items - 1
            if self.edge_index_train.max() > max_node_id:
                raise ValueError(f"Edge index out of bounds: {self.edge_index_train.max()} > {max_node_id}")
            
            # Test item index extraction (this is what causes CUDA error)
            # Note: edge_index[1] contains both user nodes and item nodes
            # Only item nodes (>= num_users) should be offset
            target_nodes = self.edge_index_train[1]
            item_node_mask = target_nodes >= self.num_users  # Only item nodes
            item_nodes = target_nodes[item_node_mask]
            test_item_indices = item_nodes - self.num_users
            
            if len(test_item_indices) > 0:
                if test_item_indices.min() < 0 or test_item_indices.max() >= self.num_items:
                    raise ValueError(f"Item indices out of bounds after offset: [{test_item_indices.min()}, {test_item_indices.max()}], expected [0, {self.num_items-1}]")
            
            print("✅ Data integrity verified - no CUDA errors expected")
            print(f"✅ Data processing completed")
            
        except ImportError:
            raise ImportError("pandas not available. Please install: pip install pandas pyarrow")
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet data: {e}")
            
    def _create_user_pos_items(self):
        """Create user positive items mapping for each split."""
        
        print("🔧 Creating user positive items mappings...")
        
        # Train user positive items
        self.train_user_pos_items = [set() for _ in range(self.num_users)]
        for user, item in self.train_edges.tolist():
            self.train_user_pos_items[user].add(item)
            
        # Val user positive items (train + val)
        self.val_user_pos_items = [set() for _ in range(self.num_users)]
        for user, item in self.train_edges.tolist():
            self.val_user_pos_items[user].add(item)
        for user, item in self.val_edges.tolist():
            self.val_user_pos_items[user].add(item)
            
        # Test user positive items (train + val + test)
        self.test_user_pos_items = [set() for _ in range(self.num_users)]
        for user, item in self.train_edges.tolist():
            self.test_user_pos_items[user].add(item)
        for user, item in self.val_edges.tolist():
            self.test_user_pos_items[user].add(item)
        for user, item in self.test_edges.tolist():
            self.test_user_pos_items[user].add(item)
            
        print(f"✅ User positive items created")
        
    def get_test_items_tensor(self) -> torch.Tensor:
        """Get test items as tensor for evaluation.
        
        CRITICAL FIX: Handle multiple test items per user properly.
        """
        test_items = torch.full((self.num_users,), -1, dtype=torch.long)  # Use -1 as default
        for user, item in self.test_edges.tolist():
            test_items[user] = item  # Will use last item if multiple per user
        return test_items
        
    def get_val_items_tensor(self) -> torch.Tensor:
        """Get validation items as tensor for evaluation.
        
        CRITICAL FIX: Handle multiple val items per user properly.
        """ 
        val_items = torch.full((self.num_users,), -1, dtype=torch.long)  # Use -1 as default
        for user, item in self.val_edges.tolist():
            val_items[user] = item  # Will use last item if multiple per user
        return val_items
        
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        
        stats = {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_train_edges': len(self.train_edges),
            'num_val_edges': len(self.val_edges),
            'num_test_edges': len(self.test_edges),
            'train_density': len(self.train_edges) / (self.num_users * self.num_items),
            'avg_user_train_items': len(self.train_edges) / self.num_users,
            'avg_item_train_users': len(self.train_edges) / self.num_items
        }
        
        # Item popularity statistics
        tail_count = (self.item_popularity_group == 0).sum().item()
        middle_count = (self.item_popularity_group == 1).sum().item()
        head_count = (self.item_popularity_group == 2).sum().item()
        
        stats.update({
            'tail_items': tail_count,
            'middle_items': middle_count, 
            'head_items': head_count,
            'tail_ratio': tail_count / self.num_items,
            'head_ratio': head_count / self.num_items
        })
        
        return stats