#!/usr/bin/env python3
"""
Graph Augmentation Experiments Runner
Runs experiments comparing different graph augmentation strategies with train/val/test evaluation.
"""

import os
import sys
import datetime
import time
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.config import get_device, set_seed, PROJECT_ROOT as CONFIG_ROOT
from src.data_loader import DataProcessor
from src.ile_training import train_model_with_ile, save_final_model
from src.graph_augmentation import GraphAugmentation


def run_augmentation_experiment(config_name: str, lambda_ile: float, dropout_type: str, device: torch.device, data_processor: 'DataProcessor'):
    """Run single augmentation experiment with proper integration."""
    
    print(f"\n🔄 Starting {config_name}")
    print(f"   λ_ILE: {lambda_ile}")
    print(f"   Dropout: {dropout_type}")
    
    # CRITICAL FIX: Input validation
    if not config_name or not isinstance(config_name, str):
        raise ValueError(f"Invalid config_name: {config_name}")
    
    if not isinstance(lambda_ile, (int, float)) or lambda_ile < 0:
        raise ValueError(f"Invalid lambda_ile: {lambda_ile}")
    
    if dropout_type is not None and dropout_type not in ['uniform', 'degree_aware']:
        raise ValueError(f"Invalid dropout_type: {dropout_type}")
    
    if data_processor is None:
        raise ValueError("data_processor cannot be None")
    
    # Validate data processor state
    required_attrs = [
        'num_users', 'num_items', 'train_edges', 'edge_index_train',
        'val_edges', 'test_edges', 'train_user_pos_items', 
        'val_user_pos_items', 'test_user_pos_items', 
        'item_degree', 'item_popularity_group'
    ]
    
    for attr in required_attrs:
        if not hasattr(data_processor, attr):
            raise ValueError(f"Missing required attribute in data_processor: {attr}")
        
        value = getattr(data_processor, attr)
        if value is None:
            raise ValueError(f"Attribute {attr} is None")
    
    # Validate data dimensions
    if data_processor.num_users <= 0 or data_processor.num_items <= 0:
        raise ValueError(f"Invalid data dimensions: users={data_processor.num_users}, items={data_processor.num_items}")
    
    if len(data_processor.train_edges) == 0:
        raise ValueError("Training edges are empty")
    
    # No data loading - use pre-loaded data
    print(f"📊 Using pre-loaded data: {data_processor.num_users} users, {data_processor.num_items} items")
    
    # Train model with augmentation via ile_training
    print(f"🚀 Training {config_name}...")
    
    try:
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
            lambda_ile=lambda_ile,
            model_name=config_name,
            save_checkpoints=True,
            dropout_type=dropout_type  # Pass augmentation config
        )
    except Exception as e:
        print(f"❌ Training failed for {config_name}: {e}")
        raise
    
    # CRITICAL FIX: Validate training results
    if model is None:
        raise ValueError(f"Training returned None model for {config_name}")
    
    if not isinstance(metrics, dict):
        raise ValueError(f"Training returned invalid metrics type: {type(metrics)}")
    
    # Validate required metrics exist
    required_test_metrics = ['test_Recall@20', 'test_TailRecall@20', 'test_Coverage@20']
    required_val_metrics = ['val_Recall@20', 'val_TailRecall@20', 'val_Coverage@20']
    required_training_metrics = ['best_epoch', 'total_epochs']
    
    missing_metrics = []
    for metric in required_test_metrics + required_val_metrics + required_training_metrics:
        if metric not in metrics:
            missing_metrics.append(metric)
    
    if missing_metrics:
        print(f"⚠️  Warning: Missing metrics for {config_name}: {missing_metrics}")
        # Fill with default values
        for metric in missing_metrics:
            if 'epoch' in metric:
                metrics[metric] = 0
            else:
                metrics[metric] = 0.0
    
    # Save final model with error handling
    try:
        model_path = save_final_model(
            model=model,
            model_name=config_name,
            metadata=metrics
        )
    except Exception as e:
        print(f"⚠️  Warning: Failed to save model for {config_name}: {e}")
        model_path = Path("model_save_failed.pt")
    
    # Extract key metrics for reporting (using test metrics) with safe access
    eval_results = {
        'model_name': config_name,
        'lambda_ile': float(lambda_ile),
        'dropout_type': str(dropout_type) if dropout_type is not None else 'None',
        'timestamp': datetime.datetime.now().isoformat(),
        
        # Test set metrics (primary) with safe access
        'Recall@20': float(metrics.get('test_Recall@20', 0.0)),
        'TailRecall@20': float(metrics.get('test_TailRecall@20', 0.0)),
        'Coverage@20': float(metrics.get('test_Coverage@20', 0.0)),
        'NDCG@20': float(metrics.get('test_NDCG@20', 0.0)),
        
        # Validation metrics with safe access
        'val_Recall@20': float(metrics.get('val_Recall@20', 0.0)),
        'val_TailRecall@20': float(metrics.get('val_TailRecall@20', 0.0)),
        'val_Coverage@20': float(metrics.get('val_Coverage@20', 0.0)),
        
        # Training info with safe access
        'best_epoch': int(metrics.get('best_epoch', 0)),
        'total_epochs': int(metrics.get('total_epochs', 0)),
        'model_path': str(model_path)
    }
    
    # CRITICAL FIX: Validate result values are finite
    for key, value in eval_results.items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            print(f"⚠️  Warning: Non-finite value for {key}: {value}, setting to 0")
            if isinstance(value, int):
                eval_results[key] = 0
            else:
                eval_results[key] = 0.0
    
    return eval_results


def save_results(results_list: list, filename: str = None):
    """Save results to CSV file."""
    
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"augmentation_results_{timestamp}.csv"
    
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    filepath = results_dir / filename
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"💾 Results saved to: {filepath}")
    
    return filepath


def print_results_summary(results_list: list):
    """Print formatted results summary."""
    
    print(f"\n📊 AUGMENTATION EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\n🔄 Augmentation Results:")
    print(f"{'Model':<35} {'Recall@20':<12} {'TailRecall@20':<15} {'Coverage@20':<12}")
    print("-"*80)
    
    for result in results_list:
        model_name = result['model_name']
        recall = result['Recall@20']
        tail_recall = result['TailRecall@20']
        coverage = result['Coverage@20']
        
        print(f"{model_name:<35} {recall:<12.4f} {tail_recall:<15.4f} {coverage:<12.4f}")


def cleanup_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(data_processor=None):
    """Main augmentation experiments runner."""
    
    print("="*80)
    print("🔄 GRAPH AUGMENTATION EXPERIMENTS")
    print("="*80)
    
    try:
        # Setup with validation
        set_seed()
        device = get_device()
        
        print(f"Device: {device}")
        
        # Use provided data processor or load new one
        if data_processor is None:
            print("📁 Loading data (no pre-loaded data provided)...")
            try:
                data_processor = DataProcessor()
            except Exception as e:
                print(f"❌ Failed to load data: {e}")
                return None
        else:
            print("📊 Using pre-loaded data processor")
            
            # CRITICAL FIX: Validate pre-loaded data processor
            try:
                # Basic validation
                if not hasattr(data_processor, 'num_users'):
                    raise ValueError("data_processor missing num_users")
                if not hasattr(data_processor, 'num_items'):  
                    raise ValueError("data_processor missing num_items")
                
                print(f"   Users: {data_processor.num_users}")
                print(f"   Items: {data_processor.num_items}")
                
            except Exception as e:
                print(f"❌ Invalid data processor: {e}")
                return None
        
        # CRITICAL FIX: Validate experiment configurations
        experiment_configs = [
            {
                'name': 'LightGCN + ILE',
                'lambda_ile': 1.0,
                'dropout_type': None  # No augmentation, just ILE
            },
            {
                'name': 'LightGCN + UniformDropout',
                'lambda_ile': 0.0,
                'dropout_type': 'uniform'
            },
            {
                'name': 'LightGCN + DegreeDropout', 
                'lambda_ile': 0.0,
                'dropout_type': 'degree_aware'
            }
        ]
        
        # Validate each configuration
        valid_configs = []
        for i, config in enumerate(experiment_configs):
            try:
                # Validate config structure
                if not isinstance(config, dict):
                    raise ValueError(f"Config {i} is not a dictionary")
                
                required_keys = ['name', 'lambda_ile', 'dropout_type']
                missing_keys = [key for key in required_keys if key not in config]
                if missing_keys:
                    raise ValueError(f"Config {i} missing keys: {missing_keys}")
                
                # Validate config values
                if not config['name'] or not isinstance(config['name'], str):
                    raise ValueError(f"Invalid name in config {i}: {config['name']}")
                
                if not isinstance(config['lambda_ile'], (int, float)) or config['lambda_ile'] < 0:
                    raise ValueError(f"Invalid lambda_ile in config {i}: {config['lambda_ile']}")
                
                dropout_type = config['dropout_type']
                if dropout_type is not None and dropout_type not in ['uniform', 'degree_aware']:
                    raise ValueError(f"Invalid dropout_type in config {i}: {dropout_type}")
                
                valid_configs.append(config)
                
            except Exception as e:
                print(f"⚠️  Warning: Skipping invalid config {i}: {e}")
                continue
        
        if not valid_configs:
            print("❌ No valid experiment configurations")
            return None
        
        experiment_configs = valid_configs
        print(f"Number of experiments: {len(experiment_configs)}")
        
        # Run experiments
        results_list = []
        failed_experiments = []
        
        for i, exp_config in enumerate(experiment_configs):
            try:
                print(f"\n--- Experiment {i+1}/{len(experiment_configs)} ---")
                
                # CRITICAL FIX: Additional config validation before running
                config_name = exp_config['name']
                lambda_ile = exp_config['lambda_ile']
                dropout_type = exp_config['dropout_type']
                
                print(f"Running: {config_name}")
                print(f"   λ_ILE: {lambda_ile}")
                print(f"   Dropout: {dropout_type}")
                
                # Run experiment
                result = run_augmentation_experiment(
                    config_name=config_name,
                    lambda_ile=lambda_ile,
                    dropout_type=dropout_type,
                    device=device,
                    data_processor=data_processor
                )
                
                if result is None:
                    raise ValueError("Experiment returned None result")
                
                # CRITICAL FIX: Validate result structure
                required_keys = ['model_name', 'lambda_ile', 'dropout_type', 'Recall@20', 'TailRecall@20', 'Coverage@20']
                missing_keys = [key for key in required_keys if key not in result]
                if missing_keys:
                    raise ValueError(f"Missing keys in result: {missing_keys}")
                
                results_list.append(result)
                
                # Print intermediate result with safe access
                recall = result.get('Recall@20', 0.0)
                tail_recall = result.get('TailRecall@20', 0.0)
                coverage = result.get('Coverage@20', 0.0)
                
                print(f"✅ Results for {config_name}:")
                print(f"   Recall@20: {recall:.4f}")
                print(f"   TailRecall@20: {tail_recall:.4f}")
                print(f"   Coverage@20: {coverage:.4f}")
                
                # Cleanup memory with error handling
                try:
                    cleanup_memory()
                except Exception as cleanup_error:
                    print(f"⚠️  Warning: Memory cleanup failed: {cleanup_error}")
                
            except KeyboardInterrupt:
                print(f"\n⏸️  Experiments interrupted by user at {exp_config.get('name', 'unknown')}")
                break
                
            except Exception as e:
                exp_name = exp_config.get('name', 'unknown')
                print(f"❌ Error in experiment {exp_name}: {e}")
                failed_experiments.append({'name': exp_name, 'error': str(e)})
                
                # Continue with next experiment instead of stopping
                continue
        
        # CRITICAL FIX: Results validation and reporting
        if results_list:
            try:
                # Validate results before saving
                valid_results = []
                for result in results_list:
                    if isinstance(result, dict) and 'model_name' in result:
                        valid_results.append(result)
                    else:
                        print(f"⚠️  Warning: Skipping invalid result: {result}")
                
                if valid_results:
                    filepath = save_results(valid_results)
                    print_results_summary(valid_results)
                    
                    print(f"\n💾 All results saved to: {filepath}")
                    
                    # Report success/failure summary
                    successful_count = len(valid_results)
                    failed_count = len(failed_experiments)
                    total_count = len(experiment_configs)
                    
                    print(f"\n📊 EXPERIMENT SUMMARY:")
                    print(f"   Successful: {successful_count}/{total_count}")
                    print(f"   Failed: {failed_count}/{total_count}")
                    
                    if failed_experiments:
                        print(f"\n❌ Failed experiments:")
                        for failed in failed_experiments:
                            print(f"   {failed['name']}: {failed['error']}")
                    
                    if successful_count > 0:
                        print("🎉 Augmentation experiments completed!")
                        return str(filepath)
                    else:
                        print("❌ No experiments completed successfully")
                        return None
                else:
                    print("❌ No valid results to save")
                    return None
                
            except Exception as e:
                print(f"❌ Error processing results: {e}")
                return None
        else:
            print("❌ No experiments completed successfully")
            if failed_experiments:
                print(f"\n❌ All {len(failed_experiments)} experiments failed:")
                for failed in failed_experiments:
                    print(f"   {failed['name']}: {failed['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Critical error in main augmentation runner: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()