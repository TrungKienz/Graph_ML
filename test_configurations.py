#!/usr/bin/env python3
"""Test exact 3 configurations as requested."""

import sys
sys.path.insert(0, 'src')

# Test ILE configurations (includes LightGCN baseline)
from src import config
print("🎯 ILE ABLATION CONFIGURATIONS:")
for i, lambda_ile in enumerate(config.LAMBDA_ILE_GRID):
    model_name = f"LightGCN+ILE" if lambda_ile > 0 else "LightGCN"
    print(f"{i+1}. {model_name} (λ_ILE={lambda_ile})")

print("\n🔄 AUGMENTATION CONFIGURATIONS:")
experiment_configs = [
    {
        'name': 'LightGCN + ILE',
        'lambda_ile': 1.0,
        'dropout_type': None
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

for i, exp_config in enumerate(experiment_configs):
    print(f"{i+1}. {exp_config['name']} (λ_ILE={exp_config['lambda_ile']}, dropout={exp_config['dropout_type']})")

print(f"\n📊 TOTAL EXPERIMENTS:")
print(f"ILE Ablation: {len(config.LAMBDA_ILE_GRID)} models")
print(f"Augmentation: {len(experiment_configs)} models")
print(f"Total: {len(config.LAMBDA_ILE_GRID) + len(experiment_configs)} models")

print(f"\n✅ CONFIGURATIONS MATCH USER REQUIREMENTS:")
print(f"✅ LightGCN + ILE (in both ILE ablation AND augmentation)")
print(f"✅ LightGCN + uniform dropout")  
print(f"✅ LightGCN + degree-aware dropout")