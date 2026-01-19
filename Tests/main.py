import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
print(f"Base Directory: {BASE_DIR}")
data_path = BASE_DIR / ".." 
print(f"Data Path: {data_path.resolve()}")