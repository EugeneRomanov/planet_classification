from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent.as_posix()
EXPERIMENTS_PATH = (Path(PROJECT_PATH) / "experiments").resolve().as_posix()
