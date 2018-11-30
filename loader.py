#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on 2018/11/25

@author: Gong
"""

import pandas as pd
import json

CONFIG_FILE = "config.json"
MIN_COUNT = 10


def load_config(config_file=CONFIG_FILE):
    vocab, id_word, word_id = json.load(open(config_file))
    id_word = {int(i): j for i, j in id_word.items()}
    return vocab, id_word, word_id


def generate_config(input_file, config_file, min_count=MIN_COUNT):
    vocab = {}
    df = pd.read_csv(input_file, header=0, index_col=0)
    for index, row in df.iterrows():
        if (not isinstance(row["title"], str)) or (not isinstance(row["content"], str)):
            continue
        for word in row["title"]:
            vocab[word] = vocab.get(word, 0) + 1
        for word in row["content"]:
            vocab[word] = vocab.get(word, 0) + 1
    vocab = {i: j for i, j in vocab.items() if j >= min_count}
    id_word = {i + 4: j for i, j in enumerate(vocab)}
    id_word[0] = "<MASK>"
    id_word[1] = "<UNK>"
    id_word[2] = "<BEG>"
    id_word[3] = "<EOS>"
    word_id = {j: i for i, j in id_word.items()}
    json.dump([vocab, id_word, word_id], open(config_file, "w"))


if __name__ == "__main__":
    generate_config(r"data\news_test.csv", CONFIG_FILE, MIN_COUNT)
    vocab, id_word, word_id = load_config(CONFIG_FILE)
    print("# of vocab:", len(vocab))
