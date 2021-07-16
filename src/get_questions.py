import re

PATTERN = r"\?|проблем|почему|не знаю"


def is_question(df, pattern=PATTERN):
    return df[df["text"].str.contains(pattern, flags=re.IGNORECASE)]


def top_level_comments(df):
    return df[df["parent_id"].isna()][["comment_id", "step_id", "text"]]
