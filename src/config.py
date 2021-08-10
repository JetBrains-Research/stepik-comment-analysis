from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve(strict=True).parent.parent

# data dir with datasets
DATA_DIR = os.path.join(ROOT_DIR, "data")

COMMENTS_FILE_NAME = "COMMENTS.csv.gz"
STEPS_FILE_NAME = "AN-695-Egor-STEPS.csv.gz"
COURSES_FILE_NAME = "popular_courses.csv"
