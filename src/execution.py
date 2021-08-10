import numpy as np
import pandas as pd
from distance import get_similar_questions
from preprocessing import TFIDFEmbedding, BertEmbedding

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")


def find_similar_comments(df, threshold, k, embedding, return_dist=True):
    comments = df.comment_id.values
    if embedding == "bert":
        bert = BertEmbedding(df)
        vectorized_texts = bert.evaluate()
    elif embedding == "tfidf":
        tfidf = TFIDFEmbedding(df)
        vectorized_texts = tfidf.evaluate()
    else:
        raise Exception("The wrong method to get embeddings. Use 'bert' or 'tfidf'")

    top_k_index, top_k_dist = get_similar_questions(vectorized_texts, threshold, k)
    comm_idx = np.append(comments, -1)[top_k_index]

    df_similarity = pd.DataFrame(comm_idx).add_prefix("comment_")
    if return_dist:
        df_dist = pd.DataFrame(top_k_dist).add_prefix("dist_")
        df_similarity = pd.concat([df_dist, df_similarity], axis=1)
    df_similarity.insert(0, "id", comments)
    df_similarity.insert(0, "step_id", df.step_id.values)
    return df_similarity


def combine_methods(df, k, threshold_tfidf, threshold_bert, cols):
    df_tfidf = find_similar_comments(df=df, threshold=threshold_tfidf, k=k, embedding="tfidf", return_dist=False)
    df_bert = find_similar_comments(df=df, threshold=threshold_bert, k=k, embedding="bert", return_dist=False)

    df_similarity = df_bert.merge(df_tfidf, on=["id", "step_id"])
    comments = df_similarity[["step_id", "id"]].values
    sim = df_similarity.drop(["step_id", "id"], axis=1).values

    repeat_comments = np.array([]).reshape(0, 2)
    sim_comments = []
    for i, idx in enumerate(comments):
        uniq_sim = np.unique(sim[i])
        uniq_sim = uniq_sim[uniq_sim >= 0]
        split_sim = [uniq_sim[x : x + cols] for x in range(0, len(uniq_sim), cols)]
        repeat_comments = np.concatenate((repeat_comments, np.tile(idx, (len(split_sim), 1))), axis=0)
        sim_comments.extend(split_sim)

    df_similarity = pd.DataFrame(sim_comments)
    df_similarity.insert(0, "id", repeat_comments[:, 1])
    df_similarity.insert(0, "step_id", repeat_comments[:, 0])
    return df_similarity
