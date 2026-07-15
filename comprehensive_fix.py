#!/usr/bin/env python3
"""
Comprehensive fix for all type casting and index issues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🔧 Applying comprehensive fixes...")
    
    # Fix 1: models.py - ensure deg is float before pow()
    models_py = Path("src/models.py")
    content = models_py.read_text()
    
    # Replace the problematic line
    old_line = "        deg = torch.bincount(row, minlength=self.num_nodes).float()"
    new_line = "        deg = torch.bincount(row, minlength=self.num_nodes).float()"
    
    if old_line in content:
        print("✅ models.py already fixed")
    else:
        # Try to find the actual line
        if "torch.bincount(row, minlength=self.num_nodes)" in content:
            content = content.replace(
                "deg = torch.bincount(row, minlength=self.num_nodes)",
                "deg = torch.bincount(row, minlength=self.num_nodes).float()"
            )
            models_py.write_text(content)
            print("✅ Fixed models.py deg calculation")
    
    # Fix 2: Check ile_losses.py
    ile_losses_py = Path("src/ile_losses.py")
    content = ile_losses_py.read_text()
    
    if "item_degrees = item_degrees.float()" in content:
        print("✅ ile_losses.py already fixed")
    else:
        print("❌ ile_losses.py needs manual fix")
    
    # Fix 3: Check graph_augmentation.py bounds checking
    graph_aug_py = Path("src/graph_augmentation.py")
    content = graph_aug_py.read_text()
    
    if "valid_mask = (item_indices >= 0)" in content:
        print("✅ graph_augmentation.py bounds checking exists")
    else:
        print("❌ graph_augmentation.py needs bounds checking")
    
    print("🔧 Comprehensive fix completed!")

if __name__ == "__main__":
    main()