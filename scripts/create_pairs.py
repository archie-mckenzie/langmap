import csv
import json

def ensure_unique_items(translations):
    seen = set()
    for key, phrases in translations.items():
        translations[key] = [phrase for phrase in phrases if not (phrase in seen or seen.add(phrase))]
    return translations

def main(dataset_names):

    candidates = {}

    for i, name in enumerate(dataset_names):
        with open(f'data/raw/{name}.tsv', 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if i == 0:
                    candidates[row[1]] = [row[3]]
                elif row[1] in candidates and len(candidates[row[1]]) == i:
                    candidates[row[1]].append(row[3])
    
    filtered_dict = {k: v for k, v in ensure_unique_items(candidates).items() if isinstance(v, list) and len(v) == len(dataset_names)}
    filtered_array = [v + [k] for (k, v) in filtered_dict.items()]
    print(filtered_array)
    print(len(filtered_array))

    with open('data/curated/sentences.json', 'w') as file:
        json.dump(filtered_array, file)

if __name__ == '__main__':
    DATASET_NAMES = ['fr', 'es', 'de', 'zh', 'ja', 'ru', 'pt']
    main(DATASET_NAMES)
