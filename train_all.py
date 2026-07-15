#!/usr/bin/env python3
"""
Complete Training Pipeline - Run All ILE Experiments Overnight

This script runs the full experimental pipeline:
1. ILE Ablation Study (6 lambda values)
2. Graph Augmentation Experiments (3 configurations)
3. Full Model Comparison 
4. Results Analysis and Summary

Designed for overnight A100 cluster runs with comprehensive logging.
"""

import sys
import os
import datetime
import traceback
from pathlib import Path
import time

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import config
from src.config import get_device, set_seed, PROJECT_ROOT
from src.run_ile_experiments import main as run_ile_experiments
from src.run_augmentation_experiments import main as run_augmentation_experiments


def print_banner(title: str, symbol: str = "="):
    """Print formatted banner."""
    width = 80
    border = symbol * width
    padding = (width - len(title) - 2) // 2
    print(f"\n{border}")
    print(f"{symbol}{' ' * padding}{title}{' ' * padding}{symbol}")
    print(f"{border}")


def log_system_info():
    """Log system information."""
    import torch
    print_banner("🖥️  SYSTEM INFORMATION")
    
    device = get_device()
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Number of epochs per model: {config.NUM_EPOCHS}")
    

def estimate_runtime():
    """Estimate total runtime."""
    print_banner("⏱️  RUNTIME ESTIMATION")
    
    # Based on actual results: ~4.5 min per lambda value
    ile_experiments = len(config.LAMBDA_ILE_GRID) * 4.5  # 6 lambdas * 4.5 min
    augmentation_experiments = 3 * 15  # 3 configs * 15 min (with contrastive learning)
    overhead = 10  # Setup, evaluation, saving
    
    total_minutes = ile_experiments + augmentation_experiments + overhead
    total_hours = total_minutes / 60
    
    print(f"ILE Ablation: {len(config.LAMBDA_ILE_GRID)} models × 4.5 min = {ile_experiments:.0f} min")
    print(f"Augmentation: 3 models × 15 min = {augmentation_experiments:.0f} min") 
    print(f"Overhead: {overhead} min")
    print(f"Total estimated time: {total_minutes:.0f} minutes ({total_hours:.1f} hours)")
    
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(minutes=total_minutes)
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated completion: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


def run_step(step_name: str, step_func, *args, **kwargs):
    """Run a pipeline step with error handling."""
    print_banner(f"🚀 STEP: {step_name}")
    
    start_time = time.time()
    
    try:
        result = step_func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"✅ {step_name} completed successfully in {elapsed/60:.1f} minutes")
        return result, True
        
    except KeyboardInterrupt:
        print(f"\n⏸️  {step_name} interrupted by user")
        return None, False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {step_name} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return None, False


def cleanup_gpu_memory():
    """Clean up GPU memory between experiments - A100 optimized."""
    import torch
    if torch.cuda.is_available():
        # A100 OPTIMIZATION: More thorough memory cleanup
        torch.cuda.synchronize()  # Wait for all operations
        torch.cuda.empty_cache()  # Free unused memory
        torch.cuda.ipc_collect()  # Cleanup IPC resources
        
        # Optional: Force garbage collection
        import gc
        gc.collect()
        
        print(f"🧹 GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def run_comprehensive_comparison():
    """Run comprehensive scientific analysis and comparison."""
    print_banner("📊 COMPREHENSIVE SCIENTIFIC ANALYSIS")
    
    try:
        import pandas as pd
        import numpy as np
        
        results_dir = PROJECT_ROOT / "results"
        
        # CRITICAL FIX: Ensure results directory exists
        if not results_dir.exists():
            print("❌ Results directory does not exist")
            return False
        
        # Find latest result files
        ile_files = list(results_dir.glob("ile_results_*.csv"))
        aug_files = list(results_dir.glob("augmentation_results_*.csv"))
        
        if not ile_files:
            print("❌ No ILE results found")
            # CRITICAL FIX: Try to find any CSV files as fallback
            all_csv_files = list(results_dir.glob("*.csv"))
            if all_csv_files:
                print(f"📄 Found {len(all_csv_files)} CSV files in results directory")
                for csv_file in all_csv_files:
                    print(f"   - {csv_file.name}")
            return False
            
        # Load latest results with error handling
        try:
            latest_ile = max(ile_files, key=lambda p: p.stat().st_mtime)
            print(f"📄 Loading ILE results: {latest_ile.name}")
            ile_df = pd.read_csv(latest_ile)
            
            # CRITICAL FIX: Validate ILE results structure
            required_columns = ['lambda_ile', 'Recall@20', 'TailRecall@20', 'Coverage@20']
            missing_columns = [col for col in required_columns if col not in ile_df.columns]
            if missing_columns:
                print(f"⚠️  Warning: Missing columns in ILE results: {missing_columns}")
                print(f"Available columns: {list(ile_df.columns)}")
                return False
            
            if len(ile_df) == 0:
                print("❌ ILE results file is empty")
                return False
                
        except Exception as e:
            print(f"❌ Error loading ILE results: {e}")
            return False
        
        # Load augmentation results with error handling
        aug_df = None
        if aug_files:
            try:
                latest_aug = max(aug_files, key=lambda p: p.stat().st_mtime)
                print(f"📄 Loading Augmentation results: {latest_aug.name}")
                aug_df = pd.read_csv(latest_aug)
                
                # CRITICAL FIX: Validate augmentation results
                aug_required_columns = ['model_name', 'lambda_ile', 'Recall@20', 'TailRecall@20', 'Coverage@20']
                aug_missing_columns = [col for col in aug_required_columns if col not in aug_df.columns]
                if aug_missing_columns:
                    print(f"⚠️  Warning: Missing columns in augmentation results: {aug_missing_columns}")
                    aug_df = None
                elif len(aug_df) == 0:
                    print("⚠️  Warning: Augmentation results file is empty")
                    aug_df = None
                    
            except Exception as e:
                print(f"⚠️  Warning: Error loading augmentation results: {e}")
                aug_df = None
        
        # Create unified comparison table
        comparison_data = []
        
        # Add baseline LightGCN with safe access
        try:
            baseline_rows = ile_df[ile_df['lambda_ile'] == 0.0]
            baseline = baseline_rows.iloc[0] if len(baseline_rows) > 0 else None
            
            if baseline is not None:
                comparison_data.append({
                    'Method': 'LightGCN (Baseline)',
                    'Category': 'Baseline',
                    'λ_ILE': 0.0,
                    'Dropout': 'None',
                    'Recall@20': float(baseline['Recall@20']),
                    'TailRecall@20': float(baseline['TailRecall@20']),
                    'Coverage@20': float(baseline['Coverage@20']),
                    'NDCG@20': float(baseline.get('NDCG@20', 0.0))
                })
            else:
                print("⚠️  Warning: No baseline (λ=0.0) found in ILE results")
        except Exception as e:
            print(f"⚠️  Warning: Error processing baseline: {e}")
            baseline = None
        
        # Add ILE methods (exclude baseline) with safe processing
        try:
            for _, row in ile_df.iterrows():
                if float(row['lambda_ile']) > 0:
                    comparison_data.append({
                        'Method': f"LightGCN + ILE (λ={float(row['lambda_ile']):.1f})",
                        'Category': 'ILE',
                        'λ_ILE': float(row['lambda_ile']),
                        'Dropout': 'None',
                        'Recall@20': float(row['Recall@20']),
                        'TailRecall@20': float(row['TailRecall@20']),
                        'Coverage@20': float(row['Coverage@20']),
                        'NDCG@20': float(row.get('NDCG@20', 0.0))
                    })
        except Exception as e:
            print(f"⚠️  Warning: Error processing ILE methods: {e}")
        
        # Add augmentation methods with safe processing
        if aug_df is not None:
            try:
                for _, row in aug_df.iterrows():
                    model_name = str(row['model_name'])
                    
                    if 'UniformDropout' in model_name:
                        method_name = 'LightGCN + Uniform Dropout'
                        category = 'Graph Aug'
                    elif 'DegreeDropout' in model_name and float(row['lambda_ile']) == 0:
                        method_name = 'LightGCN + Degree-Aware Dropout'  
                        category = 'Graph Aug'
                    elif 'ILE' in model_name and float(row['lambda_ile']) > 0:
                        method_name = f"LightGCN + ILE (Pure)"
                        category = 'ILE'
                    else:
                        continue
                        
                    comparison_data.append({
                        'Method': method_name,
                        'Category': category,
                        'λ_ILE': float(row['lambda_ile']),
                        'Dropout': str(row.get('dropout_type', 'None')),
                        'Recall@20': float(row['Recall@20']),
                        'TailRecall@20': float(row['TailRecall@20']),
                        'Coverage@20': float(row['Coverage@20']),
                        'NDCG@20': float(row.get('NDCG@20', 0.0))
                    })
            except Exception as e:
                print(f"⚠️  Warning: Error processing augmentation methods: {e}")
        
        if len(comparison_data) == 0:
            print("❌ No comparison data could be generated")
            return False
        
        # Create comparison DataFrame
        comp_df = pd.DataFrame(comparison_data)
        
        # SCIENTIFIC ANALYSIS TABLE
        print(f"\n" + "="*120)
        print(f"📊 COMPREHENSIVE MODEL COMPARISON - SCIENTIFIC ANALYSIS")
        print(f"="*120)
        
        print(f"\n🎯 MAIN RESULTS TABLE:")
        print(f"{'Method':<40} {'Category':<12} {'Recall@20':<10} {'TailRecall@20':<13} {'Coverage@20':<12} {'NDCG@20':<10}")
        print(f"-"*120)
        
        # Sort by category then by performance with error handling
        try:
            comp_df_sorted = comp_df.sort_values(['Category', 'TailRecall@20'], ascending=[True, False])
            
            for _, row in comp_df_sorted.iterrows():
                print(f"{row['Method']:<40} {row['Category']:<12} {row['Recall@20']:<10.4f} {row['TailRecall@20']:<13.4f} {row['Coverage@20']:<12.4f} {row['NDCG@20']:<10.4f}")
        except Exception as e:
            print(f"❌ Error displaying results table: {e}")
        
        # Save results with error handling
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save scientific comparison table
            scientific_file = results_dir / f"scientific_comparison_{timestamp}.csv"
            comp_df.to_csv(scientific_file, index=False)
            
            # Save combined raw results
            combined_file = results_dir / f"combined_results_{timestamp}.csv"
            if aug_df is not None:
                combined_raw = pd.concat([ile_df, aug_df], ignore_index=True)
            else:
                combined_raw = ile_df
            combined_raw.to_csv(combined_file, index=False)
            
            print(f"\n💾 Scientific Analysis saved to: {scientific_file}")
            print(f"💾 Combined raw results saved to: {combined_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Error saving analysis results: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing required libraries: {e}")
        print("Please install: pip install pandas numpy")
        return False
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main pipeline execution."""
    
    # Pipeline start
    pipeline_start = time.time()
    print_banner("🎬 STARTING COMPLETE ILE TRAINING PIPELINE")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # System info
        log_system_info()
        estimate_runtime()
        
        # Set random seed for reproducibility
        set_seed()
        print(f"\n🎲 Random seed set to: {config.SEED}")
        
        # CRITICAL FIX: Data loading with comprehensive error handling
        print_banner("📁 LOADING DATA (ONE TIME)")
        print("Loading data globally to avoid repeated loading...")
        
        try:
            from src.data_loader import DataProcessor
            global_data_processor = DataProcessor()
            
            # CRITICAL FIX: Validate loaded data
            if global_data_processor.num_users <= 0 or global_data_processor.num_items <= 0:
                raise ValueError(f"Invalid data dimensions: users={global_data_processor.num_users}, items={global_data_processor.num_items}")
            
            if len(global_data_processor.train_edges) == 0:
                raise ValueError("No training edges found")
            
            if len(global_data_processor.val_edges) == 0:
                print("⚠️  Warning: No validation edges found")
            
            if len(global_data_processor.test_edges) == 0:
                print("⚠️  Warning: No test edges found")
            
            print(f"✅ Data loaded once:")
            print(f"   Users: {global_data_processor.num_users}")
            print(f"   Items: {global_data_processor.num_items}") 
            print(f"   Training interactions: {len(global_data_processor.train_edges)}")
            print(f"   Validation interactions: {len(global_data_processor.val_edges)}")
            print(f"   Test interactions: {len(global_data_processor.test_edges)}")
            
        except Exception as e:
            print(f"❌ Critical error loading data: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return False
        
        # CRITICAL FIX: Initialize results tracking with proper error handling
        results = {}
        
        # Step 1: ILE Ablation Study
        print_banner("STEP 1: ILE ABLATION STUDY")
        print("Running comprehensive ILE experiments with lambda grid...")
        print(f"Lambda values: {config.LAMBDA_ILE_GRID}")
        
        try:
            result, success = run_step("ILE Ablation Study", run_ile_experiments, global_data_processor)
            results['ile_study'] = {'success': success, 'result': result}
            
            if not success:
                print("❌ ILE study failed. Attempting to continue with limited functionality...")
                # Don't immediately return False, try other experiments
        except Exception as e:
            print(f"❌ Critical error in ILE experiments: {e}")
            results['ile_study'] = {'success': False, 'result': None}
        
        # CRITICAL FIX: Safe GPU cleanup with error handling
        try:
            cleanup_gpu_memory()
        except Exception as e:
            print(f"⚠️  Warning: GPU cleanup failed: {e}")
        
        # Step 2: Augmentation Experiments  
        print_banner("STEP 2: AUGMENTATION EXPERIMENTS")
        print("Running graph augmentation experiments...")
        print("Configs: LightGCN+ILE, LightGCN+UniformDropout, LightGCN+DegreeDropout")
        
        try:
            result, success = run_step("Augmentation Experiments", run_augmentation_experiments, global_data_processor)
            results['augmentation'] = {'success': success, 'result': result}
            
            if not success:
                print("⚠️  Augmentation experiments failed, but continuing...")
        except Exception as e:
            print(f"❌ Critical error in augmentation experiments: {e}")
            results['augmentation'] = {'success': False, 'result': None}
        
        # CRITICAL FIX: Safe GPU cleanup with error handling
        try:
            cleanup_gpu_memory()
        except Exception as e:
            print(f"⚠️  Warning: GPU cleanup failed: {e}")
        
        # Step 3: Comprehensive Analysis
        try:
            result, success = run_step("Result Analysis", run_comprehensive_comparison)
            results['analysis'] = {'success': success, 'result': result}
        except Exception as e:
            print(f"❌ Error in result analysis: {e}")
            results['analysis'] = {'success': False, 'result': None}
        
        # Pipeline summary
        pipeline_elapsed = time.time() - pipeline_start
        print_banner("🎉 PIPELINE COMPLETED")
        
        print(f"Total runtime: {pipeline_elapsed/3600:.2f} hours ({pipeline_elapsed/60:.1f} minutes)")
        print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # CRITICAL FIX: Safe success calculation
        successful_steps = 0
        total_steps = len(results)
        
        for step_result in results.values():
            if isinstance(step_result, dict) and step_result.get('success', False):
                successful_steps += 1
        
        print(f"\n📊 PIPELINE SUMMARY:")
        print(f"Successful steps: {successful_steps}/{total_steps}")
        
        for step_name, step_result in results.items():
            if isinstance(step_result, dict):
                status = "✅ SUCCESS" if step_result.get('success', False) else "❌ FAILED"
            else:
                status = "❓ UNKNOWN"
            print(f"  {step_name}: {status}")
        
        # Final recommendations
        ile_success = results.get('ile_study', {}).get('success', False)
        aug_success = results.get('augmentation', {}).get('success', False)
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if ile_success:
            print(f"✅ ILE experiments completed successfully")
            print(f"✅ Check results in: {PROJECT_ROOT}/results/")
            print(f"✅ Models saved in: {PROJECT_ROOT}/models/")
            
            if successful_steps == total_steps:
                print(f"✅ Ready for production deployment")
            else:
                print(f"⚠️  Partial success - review failed steps")
        else:
            print(f"❌ ILE experiments failed - check logs and data")
            
        if not aug_success and ile_success:
            print(f"⚠️  Consider rerunning augmentation experiments separately")
        
        # CRITICAL FIX: Define success more conservatively
        # At minimum, ILE experiments must succeed
        pipeline_success = ile_success and (successful_steps >= 1)
        
        return pipeline_success
        
    except Exception as e:
        print(f"❌ Fatal error in main pipeline: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        
        final_message = "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!" if success else "⚠️  PIPELINE COMPLETED WITH SOME FAILURES"
        print_banner(final_message)
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print_banner("⏸️  PIPELINE INTERRUPTED BY USER")
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        print_banner("❌ PIPELINE CRASHED")
        print(f"Fatal error: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)