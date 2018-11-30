#!/usr/bin/env python
# -*- coding:utf-8 -*-

from bs4 import BeautifulSoup
from urllib import request


def download(title, url):
    req = request.Request(url)
    response = request.urlopen(req)
    response = response.read().decode('utf-8')
    soup = BeautifulSoup(response, 'lxml')
    tag = soup.find('div', class_='sm-article-content')
    if tag == None:
        return 0
    title = title.replace(':', '')
    title = title.replace('"', '')
    title = title.replace('|', '')
    title = title.replace('/', '')
    title = title.replace('\\', '')
    title = title.replace('*', '')
    title = title.replace('<', '')
    title = title.replace('>', '')
    title = title.replace('?', '')
    with open(r'\UC_news\society\\' + title + '.txt', 'w', encoding='utf-8') as file_object:
        file_object.write('\t\t\t\t')
        file_object.write(title)
        file_object.write('\n')
        file_object.write('该新闻地址：')
        file_object.write(url)
        file_object.write('\n')
        file_object.write(tag.get_text())
    # print('正在爬取')


if __name__ == '__main__':
    for i in range(0, 7):
        url = 'https://news.uc.cn/c_shehui/'
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.91 Safari/537.36",
                   "cookie": "sn=3957284397500558579; _uc_pramas=%7B%22fr%22%3A%22pc%22%7D"}
        res = request.Request(url, headers=headers)
        res = request.urlopen(url)
        req = res.read().decode('utf-8')
        soup = BeautifulSoup(req, 'lxml')
        print(soup.prettify())
        tag = soup.find_all('div', class_='txt-area-title')
        print(tag.name)
        # for x in tag:
        #     news_url = 'https://news.uc.cn' + x.a.get('href')
        #     print(x.a.string, news_url)
        #     download(x.a.string, news_url)
