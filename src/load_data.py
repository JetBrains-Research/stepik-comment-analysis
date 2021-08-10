import re
import pandas as pd
import numpy as np

pattern_courses = r"код|компьютерн|математи|информатик|программир|алгебр|физик|хими|unix|python|excel|linux"


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, compression="gzip")
    df = df[df.text.notna()].reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    return df


def load_courses(filepath_courses):
    df_courses = pd.read_csv(filepath_courses)
    df_courses = df_courses[df_courses.language == "ru"]
    df_courses["use_problem"] = np.logical_or.reduce(
        [
            df_courses[c].str.contains(pattern_courses, flags=re.IGNORECASE).fillna(False)
            for c in ["title", "description", "requirements", "summary", "target_audience"]
        ]
    )
    return df_courses[["course_id", "use_problem"]].reset_index(drop=True)
