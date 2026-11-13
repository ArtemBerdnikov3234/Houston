"""Central filesystem paths used by project modules."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
EVALUATION_DIR = DATA_DIR / "evaluation"
CACHE_DIR = PROJECT_ROOT / ".fastembed_cache"
QDRANT_STORAGE_DIR = PROJECT_ROOT / "qdrant_storage"

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, BENCHMARKS_DIR, EVALUATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)
