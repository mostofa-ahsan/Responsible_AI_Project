import sys
from pathlib import Path

# Stage scripts import each other as sibling modules (python src/<stage>.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
