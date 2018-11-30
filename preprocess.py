#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on 2018/11/24

@author: Gong
"""

import os
import re
import xml.sax
import jieba
import pandas as pd
from zhon.hanzi import punctuation

TEST = 0

STOPWORD_LIST = ('"', "(", ")", "[", "]", "\ue40c",
                 ",", " ", "/", "\\", "nhk", "~", "@", "#", "$", "%", "^", "&", "*",
                 "<", ":", ";", ">", "|", "[", "]")


def convert_str(input_str):
    output_str = ""
    for c in input_str:
        inside_code = ord(c)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        output_str += chr(inside_code)
    return output_str


def count_news(input_file):
    with open(input_file, "r", encoding="ansi") as f:
        return len(f.readlines()) // 6


def convert_csv(src):
    target = src[:src.find(".dat")] + ".csv"
    # if os.path.exists(target):
    #     return target
    f_in = open(src, "r", encoding="ansi")
    news = {"title": [], "content": []}
    count = 0
    while True:
        line = f_in.readline()
        if not line:
            break
        else:
            if line.startswith("<contenttitle>"):
                s_tag = "<contenttitle>"
                e_tag = "</contenttitle>"
                tag = "title"
            elif line.startswith("<content>"):
                s_tag = "<content>"
                e_tag = "</content>"
                tag = "content"
            else:
                continue
            content = line[len(s_tag): line.index(
                e_tag)].lower()
            content = re.sub(r"[{}]".format(punctuation),
                             "", str(content))
            content = convert_str(content)
            content = re.sub(
                u"\\(.*?\\)|\\{.*?}|\\[.*?]\\【.*?】", "", str(content))
            # content_cut = jieba.lcut(content)
            sentence = []
            for part in content:
                if part not in STOPWORD_LIST and part not in ("\t", " "):
                    sentence.append(part)
            count += 1 / 2
            if count % 10000 == 0:
                print("{:.2f}%".format(count / news_number * 100))
            # news[tag].append(" ".join(sentence))
            news[tag].append("".join(sentence))
    f_in.close()
    df = pd.DataFrame(news)
    df.to_csv(target)


def convert_xml(src):
    f_in = open(src, "r", encoding="ansi")
    target = src[:src.find(".dat")] + ".xml"
    # if os.path.exists(target):
    #     return target
    f_out = open(target, "w", encoding="utf-8")
    f_out.write("<root>\n")
    while True:
        line = f_in.readline()
        if not line:
            break
        else:
            if line.startswith("<contenttitle>"):
                s_tag = "<contenttitle>"
                e_tag = "</contenttitle>"
                beg = "<title>"
                eos = "</title>"
            elif line.startswith("<content>"):
                s_tag = "<content>"
                e_tag = "</content>"
                beg = s_tag
                eos = e_tag
            else:
                continue
            content = line[len(s_tag): line.index(
                e_tag)].replace("\t", " ").lower()
            content = convert_str(content)
            content = re.sub(
                u"\\(.*?\\)|\\{.*?}|\\[.*?]\\【.*?】", "", str(content))
            content = re.sub(r"[{}]".format(punctuation),
                             " ", str(content))
            content = content.replace("(", "")
            content = content.replace('"', "")
            content_cut = jieba.lcut(content)
            f_out.write(beg)
            f_out.write(" ".join(content_cut))
            f_out.write(eos + "\n")
    f_out.write("\n</root>")
    f_in.close()
    f_out.close()
    return target


class NewsHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.title = ""
        self.content = ""
        # self.count = 0
        self.data = {"title": [], "content": []}

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        # if self.CurrentData == "doc":
        #     self.count += 1
        #     print("***NEWS_" + str(self.count) + "***")

    def endElement(self, tag):
        if self.CurrentData == "title":
            # print("Title", self.title)
            self.data["title"].append(self.title)
        elif self.CurrentData == "content":
            # print("Content", self.content)
            self.data["content"].append(self.content)
        self.CurrentData = ""

    def characters(self, content):
        if self.CurrentData == "title":
            self.title = content
        elif self.CurrentData == "content":
            self.content = content

    def get_data(self):
        return self.data


def generate_data(xml_file):
    if not os.path.exists(xml_file):
        raise IOError("No such file: " + xml_file)
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = NewsHandler()
    parser.setContentHandler(handler)
    parser.parse(xml_file)
    news_data = handler.get_data()
    df = pd.DataFrame(news_data)
    csv_file = xml_file[:xml_file.find(".xml")] + ".csv"
    df.to_csv(csv_file)


if __name__ == "__main__":
    global news_number, input_file
    # news_number = count_news(input_file)
    if TEST:
        input_file = r"data\news_test.dat"
        news_number = 200
    else:
        input_file = r"data\news_train.dat"
        news_number = 1411996
    # xml_file = convert_xml(input_file)
    # generate_data(xml_file)
    csv_file = convert_csv(input_file)
    print("csv DONE")
