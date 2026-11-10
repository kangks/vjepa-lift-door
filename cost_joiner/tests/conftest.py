import sys
from pathlib import Path

# python -I does not put the package parent on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
