# langmap

Research on embeddings in different languages for COS 597A.

# Plan

<ol>
    <li>Fetch 5000 random sentences across various languages from Tatoeba data</li>
    <li>Embed with various embedding models, including ada-002, store in Pinecone</li>
    <li>Investigate relationship between semantic meaning, language, and location in embedding space</li>
    <li>Use PCA to visualize embeddings in 3D (https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_3D.ipynb, as seen at 'Text similarity models’ in https://openai.com/blog/introducing-text-and-code-embeddings)
    </li>
    <li>Do related languages place more closely, even if the sentences have the same meanings?</li>

</ol>

# Languages

<ul>
    <li>English (original)</li>
    <li>French</li>
    <li>Spanish</li>
    <li>German</li> 
    <li>Chinese</li>
    <li>Japanese</li>
</ul>

# How To

1. Create a virtual environment:

```
python3 -m venv venv
```

2. Activate the virtual environment:
- On macOS/Linux:
  ```
  source venv/bin/activate
  ```
- On Windows:
  ```
  .\venv\Scripts\activate
  ```

3. Install the requirements:
```
pip3 install -r requirements.txt
```

4. Download the latest language pair data from <a href='https://tatoeba.org/en/downloads'>Tatoeba</a> into `data/raw`. Or, decompress `data/compressed/compressed.zip` and `data/compressed/compressed2.zip` and copy their files into `data/raw`.

5. Run `create_pairs.py` to create the example sentences in the given languages. With the data provided there should be 6696.

6. Make sure you have a `.env` file which looks like this:

```
OPENAI_API_KEY=""
PINECONE_API_KEY=""
PINECONE_ENV = ""
```

Replace the empty strings with your environment variables.

7. Run `embed_and_store` to store embeddings in Pinecone.

7. Run `langmap.py` for results.