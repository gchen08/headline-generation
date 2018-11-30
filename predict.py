#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2018/11/25

@author: Gong
"""

import pandas as pd
from keras.models import load_model


TEST_File = r"data\news_test.csv"

df = pd.read_csv(TEST_File)
content = []
title = []
for index, row in df.iterrows():
    title.append(row["title"])
    content.append(row["content"])

model = load_model("seq2seq_model.h5")
model.load_weights("best_model.weights")

for i in range(len(content)):
    print("actual title: {}\n".format(title[i]))
    print("predicted title: {}\n".format(model.predict(content[i])))

print("End.")
