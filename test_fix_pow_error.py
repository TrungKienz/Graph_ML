#!/usr/bin/env python3
"""
Test fix for pow() error with degree calculation
"""

import torch
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ile_losses import compute_degree_aware_dropout_probs

def main():
    print("🧪 Testing pow() error fix...")
    
    # Create test item degrees (as integer tensor - this was causing the error)
    item_degrees = torch.tensor([1, 5, 10, 50, 100, 500], dtype=torch.long)
    
    print(f"Item degrees: {item_degrees}")
    print(f"Item degrees dtype: {item_degrees.dtype}")
    
    try:
        # This should work now
        dropout_probs = compute_degree_aware_dropout_probs(item_degrees)
        print(f"✅ Dropout probs computed successfully: {dropout_probs}")
        print(f"Dropout probs dtype: {dropout_probs.dtype}")
        
        # Test with edge case
        zero_degrees = torch.zeros(5, dtype=torch.long)
        zero_probs = compute_degree_aware_dropout_probs(zero_degrees)
        print(f"✅ Zero degrees handled: {zero_probs}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    main()