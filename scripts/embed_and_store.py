# ----- IMPORTS ----- #

from dotenv import load_dotenv
import os
load_dotenv()

import sys

import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

# from https://docs.pinecone.io/docs/openai
import pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
)

import json

# ----- FUNCTIONS ----- #

def main(index_name, filepath, model):

    with open(filepath, 'r') as file:
        sentences = json.load(file)

    # check if index_name index already exists (only create index if not)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=model['dimensions'])
    else: 
        print(f'Looks like a pinecone index {index_name} already exists!')
        print('Are you sure you want to proceed?')
        should_continue = input('Enter \'y\' to continue or any other key to cancel: ')
        if (should_continue != 'y') and (should_continue != ' \'y\''):
            sys.exit(0)
    # connect to index
    index = pinecone.Index(index_name)

    vectors_by_sentence = []

    for i, set in enumerate(sentences):
        result = openai.embeddings.create(
            input = set,
            model = model['model']
        )
        embeds = [record.embedding for record in result.data]
        vectors_by_sentence.append(embeds)
        meta = [{'text': text, 'en': set[len(set) - 1], 'set': i} for text in set]
        to_upsert = zip([f'{i}-{j}' for j in range(len(set))], embeds, meta)
        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert))
        print(f'{i + 1} / {len(sentences)}')

    print('Writing to data/created/vectors.json...')
    with open('data/created/vectors.json', 'w') as file:
        json.dump(vectors_by_sentence, file)

if __name__ == '__main__':
    NAME = 'langmap'
    FILEPATH = 'data/curated/sentences.json'
    MODEL = {
        "model": "text-embedding-ada-002",
        "dimensions": 1536,
    }
    main(NAME, FILEPATH, MODEL)