import os
import load_data
import get_questions
from distance import get_similar_questions
from preprocessing import bert_texts, vectorize_texts
from config import DATA_DIR, COMMENTS_FILE_NAME

DATA_DIR = DATA_DIR
COMMENTS_FILE_NAME = COMMENTS_FILE_NAME
filepath = os.path.join(DATA_DIR, COMMENTS_FILE_NAME)

STEP_ID = 6532
NEIGHBORS = 5
THRESHOLD_bert = 0.4
THRESHOLD_tfidf = 0.36


def find_similar_comments(step_id: int, threshold: float, k: int, emb="tfidf"):
    df_comments = load_data.load_data(filepath)
    df_comments = get_questions.top_level_comments(df_comments)
    df_q = get_questions.is_question(df_comments)

    df = df_q[df_q.step_id == step_id].reset_index(drop=True)
    data = df.text.values
    print(data.size)

    if emb == "bert":
        vectorized_texts = bert_texts(data)
    else:
        vectorized_texts = vectorize_texts(data)

    top_k_index, top_k_dist = get_similar_questions(vectorized_texts, threshold, k)
    col_dist = [f"dist_{i}" for i in range(k)]
    col_top_k = [f"topk_{i}" for i in range(k)]
    df[col_top_k] = top_k_index
    df[col_dist] = top_k_dist

    for i, top_k in enumerate(col_top_k):
        df = df.join(df[["text", "comment_id", top_k]], on=top_k, rsuffix=f"_{i}")
    df = df.drop(col_top_k + [f"topk_{i}_{i}" for i in range(k)], axis=1)
    df["model"] = emb
    return df


def combine_methods(step_id, k, threshold_tfidf, threshold_bert):
    cols = ["step_id", "comment_id"]
    cols.extend([f"comment_id_{i}" for i in range(k)])
    df_tfidf = find_similar_comments(step_id, threshold_tfidf, k, "tfidf")
    df_bert = find_similar_comments(step_id, threshold_bert, k, "bert")
    df = df_bert[cols].merge(df_tfidf[cols], on=["step_id", "comment_id"], suffixes=("_bert", "_tfidf"))
    return df


if __name__ == "__main__":
    print(combine_methods(STEP_ID, NEIGHBORS, THRESHOLD_tfidf, THRESHOLD_bert))
