from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DOCUMENTS_FILE = DATA_DIR / "documents.json"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
RRF_K = 60
DEFAULT_TOP_K = 10
HOST = "0.0.0.0"
PORT = 8000
