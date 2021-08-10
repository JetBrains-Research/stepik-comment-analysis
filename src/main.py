import os
from bs4 import BeautifulSoup
from config import DATA_DIR, COMMENTS_FILE_NAME, COURSES_FILE_NAME
from execution import combine_methods
import load_data
import get_questions

filepath_courses = os.path.join(DATA_DIR, COURSES_FILE_NAME)
filepath = os.path.join(DATA_DIR, COMMENTS_FILE_NAME)

STEP_ID = 6532
NEIGHBORS = 5
COLS = 5
THRESHOLD_bert = 0.4
THRESHOLD_tfidf = 0.36

df_data = load_data.load_data(filepath)
# df_courses = load_data.load_courses(filepath_courses)
# df_data = df_data.merge(df_courses, on='course_id')
df_comments = df_data[df_data.step_id == STEP_ID].reset_index(drop=True)
# df_comments = get_questions.top_level_comments(df_comments)
df_comments["text"] = df_comments["text"].apply(lambda x: BeautifulSoup(x, "lxml").text)
df_q = get_questions.is_question(df_comments)

if __name__ == "__main__":
    print(combine_methods(df_q, NEIGHBORS, THRESHOLD_tfidf, THRESHOLD_bert, COLS))
