import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_similar_questions(vectors, threshold, k):

    dist = pdist(vectors, "cosine")
    dist = squareform(dist)
    np.fill_diagonal(dist, 1)

    dict_questions = dict()
    for idx, d in enumerate(dist):
        vec = np.argwhere(d < threshold).flatten()
        if vec.size:
            dict_questions[idx] = vec[:k].tolist()
    return dict_questions
