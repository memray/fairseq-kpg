import json

import datasets
import time
import os

import tqdm

print(datasets.__version__)

datasets_list = datasets.list_datasets()
len(datasets_list)
print(', '.join(dataset for dataset in datasets_list))

from datasets import load_dataset
dataset = load_dataset('glue', 'mrpc', split='train')

len(dataset)
dataset[0]
dataset.features


from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

print(tokenizer(dataset[0]['sentence1'], dataset[0]['sentence2']))
# tokenizer.decode(tokenizer(dataset[0]['sentence1'], dataset[0]['sentence2'])['input_ids'])

def encode(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')


data_dir = '/export/share/ruimeng/data/wiki/paragraph/train/AA'

texts = []
for fname in os.listdir(data_dir):
    fpath = os.path.join(data_dir, fname)
    # print(os.path.exists(fpath))
    _texts = []
    for l in open(fpath, 'r'):
        _texts.append(json.loads(l)['text'])
    texts.extend(_texts)
    # print(len(_texts))

start_time = time.time()
# print(len(texts))
for i in tqdm.tqdm(range(10)):
    print(i)
    '''
    for t in texts:
        tokenizer(t)
    '''
    tokenizer(texts)

current_time = time.time()
elapsed_time = current_time - start_time
print("Finished iterating in: " + str(float(elapsed_time)) + " seconds")

