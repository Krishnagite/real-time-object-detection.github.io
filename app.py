import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

project_dir = Path(__file__).resolve().parent / "Real-Time-Object-Detection"
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from main import main as run_app

if __name__ == "__main__":
    run_app()
