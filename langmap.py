# langmap.py
# Author: Archie McKenzie

from dotenv import load_dotenv
import os
load_dotenv()

import json

import numpy as np

# from https://docs.pinecone.io/docs/openai
import pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
)

import random

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# ----- FUNCTIONS ----- #

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

def mean(floats):
    return sum(floats) / len(floats)

# efficient calculator of symmetric similarity matrix for a given sentence
def calculate_all_similarities(vectors):
    similarities = [[] for _ in range(len(vectors))]
    for i in range(len(vectors)):
        for j in range(i, len(vectors)):
            similarity = cosine_similarity(vectors[i], vectors[j])
            similarities[i].append(similarity)
            if (i != j):
                similarities[j].append(similarity)
    return similarities 

# ----- MAIN ----- #

def main(filepath, index_name):

    with open(filepath, 'r') as file:
        vectors_by_sentence = json.load(file)

    # en, fr, es, de, zh, ja, ru, pt, in that order
    language_codes = ['fr', 'es', 'de', 'zh', 'ja', 'ru', 'pt', 'en']
    
    num_languages = len(vectors_by_sentence[0])

    """
    Average difference that being a translation makes to similarity
    For same-sentences, various languages
    """
    
    similarities_by_language = [[[] for _ in range(num_languages)] for _ in range(num_languages)]

    for vectors in vectors_by_sentence:
        similarities = calculate_all_similarities(vectors)
        print(similarities)
        for i in range(len(similarities)):
            for j in range(i, len(similarities)):
                similarities_by_language[i][j].append(similarities[i][j])
                if (i != j):
                    similarities_by_language[j][i].append(similarities[i][j])
    
    mean_similarities_by_language = [[] for _ in range(num_languages)]
    for i in range(num_languages):
        for j in range(num_languages):
            mean_similarities = mean(similarities_by_language[i][j])
            mean_similarities_by_language[i].append(mean_similarities)
    
    print(mean_similarities_by_language) # symmetric matrix

    """
    Outlier pair-count (by Pinecone query top_k = num_languages)
    When are same sentences but in different languages NOT being returned
    And what is the language-language distribution
    """

    index = pinecone.Index(index_name)

    outlier_tally = [[0 for _ in range(num_languages)] for _ in range(num_languages)]

    for i, vectors in enumerate(vectors_by_sentence):
        for j, vector in enumerate(vectors):
            result = index.query(vector, top_k=num_languages)
            for match in result.matches:
                if match.id[0:len(str(i))+1] != f"{i}-":
                    outlier_tally[j][int(match.id.split('-')[1])] += 1 
        print(f'{i + 1} / {len(vectors_by_sentence)}')

    print(outlier_tally)

    """
    
    PCA-projected 3D graphing
    - Randomly sample one example from each sentence, for e.g. 64 vectors, 8 of each
    - Category = language
    - Project and observe clustering
    https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_3D.ipynb
    """

    pca_projection_size = 250

    samples = {"languages": []}
    random_vectors = []
    # randomly sample
    for i in range(pca_projection_size):
        random_int = random.randint(0, num_languages - 1)
        random_vector = vectors_by_sentence[i][random_int]
        random_vectors.append(random_vector)
        samples["languages"].append(language_codes[random_int])

    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(random_vectors)
    samples["embed_vis"] = vis_dims.tolist()
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(projection='3d')
    cmap = plt.get_cmap("tab20")

    # Dictionary to store points for each language
    points_by_language = {}

    # Plot each sample category individually
    for i in range(num_languages):
        sub_matrix = np.array([samples["embed_vis"][j] for j, lang in enumerate(samples["languages"]) if lang == language_codes[i]])
        if len(sub_matrix) == 0:
            continue
        x, y, z = sub_matrix[:, 0], sub_matrix[:, 1], sub_matrix[:, 2]
        colors = [cmap(i/num_languages)] * len(sub_matrix)
        ax.scatter(x, y, zs=z, zdir='z', c=colors, label=language_codes[i])
        # Save points for each language
        points_by_language[language_codes[i]] = [[point[0], point[1], point[2]] for point in sub_matrix]

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(bbox_to_anchor=(1.1, 1))
    # plt.show()

    """
    Write all the data
    To be added to langmap.org
    """

    with open('data/final/mean_similarities.json', 'w') as file:
        json.dump(mean_similarities_by_language, file)

    with open('data/final/outlier_tally.json', 'w') as file:
        json.dump(outlier_tally, file)

    with open('data/final/pca_points.json', 'w') as file:
        json.dump(points_by_language, file)
    
if __name__ == '__main__':
    VECTORS_FILEPATH = 'data/created/vectors.json'
    PINECONE_INDEX = 'langmap'
    main(VECTORS_FILEPATH, PINECONE_INDEX)
