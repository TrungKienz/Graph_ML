#!/usr/bin/env python3
import sys
sys.path.append('src')

from data_loader import DataProcessor

print("Testing data loader...")
dp = DataProcessor()
print(f"item_degree dtype: {dp.item_degree.dtype}")
print(f"Success!")