# langmap.py
# Author: Archie McKenzie
import json

import numpy as np

def cosine_similarity(alpha, beta):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    alpha (array): First vector.
    beta (array): Second vector.

    Returns:
    float: Cosine similarity between alpha and beta.
    """
    # Compute the dot product between alpha and beta
    dot_product = np.dot(alpha, beta)
    # Compute the magnitude (norm) of each vector
    norm_alpha = np.linalg.norm(alpha)
    norm_beta = np.linalg.norm(beta)
    # Avoid division by zero
    if norm_alpha == 0 or norm_beta == 0:
        return 0
    # Compute cosine similarity
    similarity = dot_product / (norm_alpha * norm_beta)
    return similarity


def main(filepath):
    
    with open(filepath, 'r') as file:
        vectors_by_sentence = json.load(file)
    
    num_sentences = len(vectors_by_sentence)
    num_languages = len(vectors_by_sentence[0])

    """
    Average difference that being a translation makes to similarity
    For same-sentences
    """
    
    """
    Average difference that being a translation makes
    In general, i.e.
    Average of all en to all fr similarity, to all de similarity, to all other en similarity
    """

    """
    Outlier pair-count (by Pinecone query top_k = num_languages)
    When are same sentences but in different languages NOT being returned
    And what is the language-language distribution
    """

    """
    PCA-projected 3D graphing
    - Randomly sample one example from each sentence, for e.g. 64 vectors, 8 of each
    - Category = language
    - Project and observe clustering
    https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_3D.ipynb
    """

    """
    Write all the data in .txt report
    To be added to langmap.org
    """
    
if __name__ == '__main__':
    VECTORS_FILEPATH = '/data/created/vectors.json'
    main(VECTORS_FILEPATH)
