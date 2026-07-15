#!/usr/bin/env python3
"""
ILE Experiments Runner
Runs comprehensive Item Loss Equalization ablation study with train/val/test evaluation.
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


def run_ile_experiment(lambda_ile: float, device: torch.device, data_processor: 'DataProcessor'):
    """Run single ILE experiment with given lambda on train/val/test."""
    
    print(f"\n🔄 Starting λ_ILE = {lambda_ile}")
    
    # CRITICAL FIX: Input validation
    if lambda_ile < 0:
        raise ValueError(f"Invalid lambda_ile: {lambda_ile}")
    
    if data_processor is None:
        raise ValueError("data_processor cannot be None")
    
    # Validate data processor state
    if not hasattr(data_processor, 'num_users') or data_processor.num_users <= 0:
        raise ValueError(f"Invalid num_users: {getattr(data_processor, 'num_users', 'missing')}")
    
    if not hasattr(data_processor, 'num_items') or data_processor.num_items <= 0:
        raise ValueError(f"Invalid num_items: {getattr(data_processor, 'num_items', 'missing')}")
    
    # Set model name
    model_name = f"LightGCN+ILE" if lambda_ile > 0 else "LightGCN"
    
    # No data loading - use pre-loaded data
    print(f"📊 Using pre-loaded data: {data_processor.num_users} users, {data_processor.num_items} items")
    
    # CRITICAL FIX: Validate required attributes exist
    required_attrs = [
        'train_edges', 'edge_index_train', 'val_edges', 'test_edges',
        'train_user_pos_items', 'val_user_pos_items', 'test_user_pos_items',
        'item_degree', 'item_popularity_group'
    ]
    
    for attr in required_attrs:
        if not hasattr(data_processor, attr):
            raise ValueError(f"Missing required attribute in data_processor: {attr}")
        
        value = getattr(data_processor, attr)
        if value is None:
            raise ValueError(f"Attribute {attr} is None")
    
    # Validate edge data is not empty
    if len(data_processor.train_edges) == 0:
        raise ValueError("Training edges are empty")
    
    if data_processor.edge_index_train.size(1) == 0:
        raise ValueError("Edge index is empty")
    
    # Train model with comprehensive evaluation
    print(f"🚀 Training {model_name} (λ_ILE={lambda_ile})...")
    
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
            model_name=f"{model_name}_{lambda_ile}",
            save_checkpoints=True
        )
    except Exception as e:
        print(f"❌ Training failed for λ_ILE = {lambda_ile}: {e}")
        raise
    
    # CRITICAL FIX: Validate training results
    if model is None:
        raise ValueError(f"Training returned None model for λ_ILE = {lambda_ile}")
    
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
        print(f"⚠️  Warning: Missing metrics for λ_ILE = {lambda_ile}: {missing_metrics}")
        # Fill with default values
        for metric in missing_metrics:
            if 'epoch' in metric:
                metrics[metric] = 0
            else:
                metrics[metric] = 0.0
    
    # Save final model
    try:
        model_path = save_final_model(
            model=model,
            model_name=f"{model_name}_{lambda_ile}",
            metadata=metrics
        )
    except Exception as e:
        print(f"⚠️  Warning: Failed to save model for λ_ILE = {lambda_ile}: {e}")
        model_path = Path("model_save_failed.pt")
    
    # Extract key metrics for reporting (using test metrics) with safe access
    eval_results = {
        'model_name': model_name,
        'lambda_ile': float(lambda_ile),
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
    """Save results to CSV file with proper naming."""
    
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ile_results_{timestamp}.csv"
    
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    filepath = results_dir / filename
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"💾 ILE Results saved to: {filepath}")
    
    return filepath


def print_results_summary(results_list: list):
    """Print formatted results summary."""
    
    print(f"\n📊 EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\n🎯 ILE Ablation Results:")
    print(f"{'Lambda':<8} {'Model':<20} {'Recall@20':<12} {'TailRecall@20':<15} {'Coverage@20':<12}")
    print("-"*75)
    
    for result in results_list:
        lambda_ile = result['lambda_ile']
        model_name = result['model_name']
        recall = result['Recall@20']
        tail_recall = result['TailRecall@20']
        coverage = result['Coverage@20']
        
        print(f"{lambda_ile:<8.1f} {model_name:<20} {recall:<12.4f} {tail_recall:<15.4f} {coverage:<12.4f}")


def cleanup_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(data_processor=None):
    """Main ILE experiments runner."""
    
    print("="*80)
    print("🎯 ILE ABLATION STUDY")
    print("="*80)
    
    try:
        # Setup with validation
        set_seed()
        device = get_device()
        
        print(f"Device: {device}")
        
        # CRITICAL FIX: Validate config parameters
        if not hasattr(config, 'LAMBDA_ILE_GRID'):
            raise ValueError("LAMBDA_ILE_GRID not found in config")
        
        if not config.LAMBDA_ILE_GRID:
            raise ValueError("LAMBDA_ILE_GRID is empty")
        
        # Validate lambda values
        for lambda_val in config.LAMBDA_ILE_GRID:
            if not isinstance(lambda_val, (int, float)) or lambda_val < 0:
                raise ValueError(f"Invalid lambda value: {lambda_val}")
        
        print(f"Lambda grid: {config.LAMBDA_ILE_GRID}")
        print(f"Number of experiments: {len(config.LAMBDA_ILE_GRID)}")
        
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
        
        # Run experiments
        results_list = []
        failed_experiments = []
        
        total_experiments = len(config.LAMBDA_ILE_GRID)
        pbar = tqdm(config.LAMBDA_ILE_GRID, desc="Lambda ILE values")
        
        for i, lambda_ile in enumerate(pbar):
            try:
                # Update progress bar
                pbar.set_description(f"λ_ILE = {lambda_ile}")
                
                # CRITICAL FIX: Validate lambda value before experiment
                if not isinstance(lambda_ile, (int, float)):
                    raise ValueError(f"Invalid lambda_ile type: {type(lambda_ile)}")
                
                if lambda_ile < 0:
                    raise ValueError(f"Negative lambda_ile: {lambda_ile}")
                
                # Run experiment
                result = run_ile_experiment(lambda_ile, device, data_processor)
                
                if result is None:
                    raise ValueError("Experiment returned None result")
                
                # CRITICAL FIX: Validate result structure
                required_keys = ['model_name', 'lambda_ile', 'Recall@20', 'TailRecall@20', 'Coverage@20']
                missing_keys = [key for key in required_keys if key not in result]
                if missing_keys:
                    raise ValueError(f"Missing keys in result: {missing_keys}")
                
                results_list.append(result)
                
                # Print intermediate result with safe access
                recall = result.get('Recall@20', 0.0)
                tail_recall = result.get('TailRecall@20', 0.0)
                coverage = result.get('Coverage@20', 0.0)
                val_recall = result.get('val_Recall@20', 0.0)
                
                print(f"✅ Results for λ_ILE = {lambda_ile}:")
                print(f"   Test Recall@20: {recall:.4f}")
                print(f"   Test TailRecall@20: {tail_recall:.4f}")
                print(f"   Test Coverage@20: {coverage:.4f}")
                print(f"   Val Recall@20: {val_recall:.4f}")
                
                # Cleanup memory with error handling
                try:
                    cleanup_memory()
                except Exception as cleanup_error:
                    print(f"⚠️  Warning: Memory cleanup failed: {cleanup_error}")
                
            except KeyboardInterrupt:
                print(f"\n⏸️  Experiments interrupted by user at λ_ILE = {lambda_ile}")
                break
                
            except Exception as e:
                print(f"❌ Error in experiment λ_ILE = {lambda_ile}: {e}")
                failed_experiments.append({'lambda_ile': lambda_ile, 'error': str(e)})
                
                # Continue with next experiment instead of stopping
                continue
        
        # CRITICAL FIX: Results validation and reporting
        if results_list:
            try:
                # Validate results before saving
                valid_results = []
                for result in results_list:
                    if isinstance(result, dict) and 'lambda_ile' in result:
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
                    total_count = len(config.LAMBDA_ILE_GRID)
                    
                    print(f"\n📊 EXPERIMENT SUMMARY:")
                    print(f"   Successful: {successful_count}/{total_count}")
                    print(f"   Failed: {failed_count}/{total_count}")
                    
                    if failed_experiments:
                        print(f"\n❌ Failed experiments:")
                        for failed in failed_experiments:
                            print(f"   λ_ILE = {failed['lambda_ile']}: {failed['error']}")
                    
                    if successful_count > 0:
                        print("🎉 ILE experiments completed!")
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
                    print(f"   λ_ILE = {failed['lambda_ile']}: {failed['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Critical error in main ILE runner: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()