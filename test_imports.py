#!/usr/bin/env python3
"""Test all critical imports before production run."""

import sys
sys.path.insert(0, 'src')

try:
    from data_loader import DataProcessor
    dp = DataProcessor()
    print(f'✅ DataProcessor: {dp.num_users} users, {dp.num_items} items')
    
    from ile_training import train_model_with_ile
    print('✅ ILE training import OK')
    
    from run_ile_experiments import main as ile_main  
    print('✅ ILE experiments import OK')
    
    from run_augmentation_experiments import main as aug_main
    print('✅ Augmentation experiments import OK')
    
    import train_all
    print('✅ train_all.py import OK')
    
    print('\n🎉 ALL CRITICAL IMPORTS SUCCESSFUL!')
    print('🚀 READY FOR PRODUCTION RUN!')
    
except Exception as e:
    print(f'❌ CRITICAL ERROR: {e}')
    import traceback
    traceback.print_exc()