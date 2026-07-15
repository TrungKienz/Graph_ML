#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.data_loader import DataProcessor
    data_processor = DataProcessor()
    print(f"✅ SUCCESS: Users={data_processor.num_users}, Items={data_processor.num_items}")
except Exception as e:
    print(f"❌ ERROR: {e}")