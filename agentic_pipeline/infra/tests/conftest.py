from pathlib import Path
import sys

INFRA = Path(__file__).resolve().parents[1]
PROJECT = INFRA.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(INFRA))
