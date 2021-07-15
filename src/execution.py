import os
import load_data
import get_questions
from distance import get_similar_questions
from preprocessing import vectorize_texts
from config import DATA_DIR, COMMENTS_FILE_NAME

DATA_DIR = DATA_DIR
COMMENTS_FILE_NAME = COMMENTS_FILE_NAME
filepath = os.path.join(DATA_DIR, COMMENTS_FILE_NAME)

# add step_id,
STEP_ID = 6532
THRESHOlD = 0.25
NEIGHBORS = 5


def find_similar_comments(step_id: int, threshold: float, k: int):
    df_comments = load_data.load_data(filepath)
    df_comments = get_questions.top_level_comments(df_comments)
    df_q = get_questions.is_question(df_comments)

    df = df_q[df_q.step_id == step_id].reset_index(drop=True)
    data = df.text.values
    comment_id = df.comment_id.values
    print(data.size)
    vectorized_texts = vectorize_texts(data)
    dict_questions = get_similar_questions(vectorized_texts, threshold, k)

    dict_comment_id = dict()
    for comm, s_comm in dict_questions.items():
        dict_comment_id[comment_id[comm]] = comment_id[s_comm].tolist()
        print("\nВопрос:")
        print(data[comm])
        print("\nПохожие вопросы:")
        print(*data[s_comm], sep="\n")
    return dict_comment_id


if __name__ == "__main__":
    print(find_similar_comments(STEP_ID, THRESHOlD, NEIGHBORS))
