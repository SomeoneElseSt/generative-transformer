"""
The kaggle dataset comes one speech per year. This short script aggregates them all into a single text file to match Karpathy's video.

"""

import os
import kagglehub

# Overrides Kaggle default download directory for my own
os.environ['KAGGLEHUB_CACHE'] = "/Users/steve/Documents/Files/code/generative_transformer/dataset"
FILES_PATH = "/Users/steve/Documents/Files/code/generative_transformer/data/datasets/rtatman/state-of-the-union-corpus-1989-2017/versions/3"

if not os.path.exists("data"):
    os.makedirs("data")
    path = kagglehub.dataset_download("rtatman/state-of-the-union-corpus-1989-2017")

text = ""
for filename in sorted(os.listdir(FILES_PATH)):
    file_path = os.path.join(FILES_PATH, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            text += f.read()

if len(text) == 0:
    print("Dataset is empty! Check your downloaded files")
    exit(1)

with open("dataset.txt", "w") as out_f:
    out_f.write(text)

