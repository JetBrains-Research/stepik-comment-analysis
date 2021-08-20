import numpy as np
from scipy.spatial.distance import cdist, squareform


def get_similar_questions(questions, vectors, threshold, k):
    dist = cdist(questions, vectors, "cosine")
    dist = np.nan_to_num(dist, nan=np.inf)
    np.fill_diagonal(dist, np.inf)

    top_k_index = dist.argsort(axis=-1)[:, :k]
    top_k_dist = np.take_along_axis(dist, top_k_index, axis=-1)
    mask = top_k_dist > threshold
    top_k_dist[mask] = -1
    top_k_index[mask] = -1

    return top_k_index, top_k_dist
