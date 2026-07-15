#!/usr/bin/env python3
"""
ILE Experiments Only - Quick Runner
Run just ILE ablation study.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.run_ile_experiments import main as run_ile_experiments


def main():
    """Run ILE experiments only."""
    print("🎯 Running ILE Experiments Only")
    print("="*50)
    
    result = run_ile_experiments()
    
    if result:
        print(f"✅ ILE experiments completed successfully!")
        print(f"📄 Results saved to: {result}")
    else:
        print("❌ ILE experiments failed")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)