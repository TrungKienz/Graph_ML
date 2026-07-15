#!/usr/bin/env python3
"""
Augmentation Experiments Only - Quick Runner
Run just augmentation experiments.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.run_augmentation_experiments import main as run_augmentation_experiments


def main():
    """Run augmentation experiments only."""
    print("🔄 Running Augmentation Experiments Only")
    print("="*50)
    
    result = run_augmentation_experiments()
    
    if result:
        print(f"✅ Augmentation experiments completed successfully!")
        print(f"📄 Results saved to: {result}")
    else:
        print("❌ Augmentation experiments failed")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)