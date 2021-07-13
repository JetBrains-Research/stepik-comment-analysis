import pandas as pd
import config

DATA_DIR = config.DATA_DIR
COMMENTS_FILE_NAME = config.COMMENTS_FILE_NAME
STEPS_FILE_NAME = config.STEPS_FILE_NAME


def load_comments():
    df = pd.read_csv(DATA_DIR / COMMENTS_FILE_NAME, compression="gzip")
    df["text"] = df["text"].astype(str)
    return df


def load_steps():
    return pd.read_csv(DATA_DIR / STEPS_FILE_NAME, compression="gzip")
