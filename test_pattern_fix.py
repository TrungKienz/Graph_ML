#!/usr/bin/env python3
"""
Test script để kiểm tra fix cho glob pattern issue.
"""

from pathlib import Path

def test_checkpoint_patterns():
    """Test the pattern issue with LightGCN+ILE model names."""
    
    test_cases = [
        'LightGCN_0.0',        # Không có +, nên OK
        'LightGCN+ILE_0.1',    # Có +, có thể gây lỗi  
    ]
    
    checkpoints_dir = Path('./checkpoints')
    
    for test_name in test_cases:
        print(f"Testing checkpoint loading for model: {test_name}")
        print("-" * 50)
        
        # Old method (có lỗi với +)
        print("🔍 Testing OLD method (glob with pattern):")
        try:
            pattern = f'checkpoint_{test_name}_epoch_*'
            print(f"   Pattern: {pattern}")
            files = list(checkpoints_dir.glob(pattern + '*.pt'))
            print(f"   Result: {len(files)} files found")
            
            if files:
                for f in files[:2]:  # Show first 2
                    print(f"   Found: {f.name}")
        except Exception as e:
            print(f"   ❌ Old method failed: {e}")
        
        print()
        
        # New method (fix)
        print("✅ Testing NEW method (iterdir + startswith):")
        try:
            prefix = f'checkpoint_{test_name}_epoch_'
            print(f"   Prefix: {prefix}")
            
            checkpoint_files = []
            if checkpoints_dir.exists():
                for file in checkpoints_dir.iterdir():
                    if file.name.startswith(prefix) and file.name.endswith('.pt'):
                        checkpoint_files.append(file)
            
            print(f"   Result: {len(checkpoint_files)} files found")
            for f in checkpoint_files[:2]:  # Show first 2
                print(f"   Found: {f.name}")
                
        except Exception as e:
            print(f"   ❌ New method failed: {e}")
        
        print("=" * 60)
        print()

if __name__ == "__main__":
    test_checkpoint_patterns()