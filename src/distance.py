import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_similar_questions(vectors, threshold, k):
    dist = pdist(vectors, "cosine")
    dist = squareform(dist)
    np.fill_diagonal(dist, np.inf)

    top_k_index = dist.argsort(axis=-1)[:, :k]
    top_k_dist = np.take_along_axis(dist, top_k_index, axis=-1)
    mask = top_k_dist > threshold
    top_k_dist[mask] = -1
    top_k_index[mask] = -1

    return top_k_index, top_k_dist
