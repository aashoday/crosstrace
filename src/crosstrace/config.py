import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = ROOT / "raw"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")