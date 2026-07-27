import sys
from pathlib import Path

# Force the project root directory into sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))