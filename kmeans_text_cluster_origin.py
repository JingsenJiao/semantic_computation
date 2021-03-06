# -*- coding: utf-8 -*-
# https://blog.csdn.net/qq_29110265/article/details/90769363

import os
import re
from os import listdir
import jieba
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

all_file = listdir('***')  # 获取文件夹中所有文件名#数据集地址
outputDir = "***"  # 结果输出地址
labels = []  # 用以存储名称
corpus = []  # 空语料库
size = 200  # 测试集容量


def buildSW():
    '''停用词的过滤'''
    typetxt = open('***')  # 停用词文档地址
    texts = ['\u3000', '\n', ' ']  # 爬取的文本中未处理的特殊字符
    '''停用词库的建立'''
    for word in typetxt:
        word = word.strip()
        texts.append(word)
    return texts


def buildWB(texts):
    '''语料库的建立'''
    for i in range(0, len(all_file)):
        filename = all_file[i]
        filelabel = filename.split('.')[0]
        labels.append(filelabel)  # 名称列表
        file_add = '***' + filename  # 数据集地址
        doc = open(file_add, encoding='utf-8').read()
        data = jieba.cut(doc)  # 文本分词
        data_adj = ''
        delete_word = []
        for item in data:
            if item not in texts:  # 停用词过滤
                # value=re.compile(r'^[0-9]+$')#去除数字
                value = re.compile(r'^[\u4e00-\u9fa5]{2,}$')  # 只匹配中文2字词以上
                if value.match(item):
                    data_adj += item + ' '
            else:
                delete_word.append(item)
        corpus.append(data_adj)  # 语料库建立完成
    # print(corpus)
    return corpus


def countIdf(corpus):
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # word=vectorizer.get_feature_names()#获取词袋模型中的所有词
    # for j in range(len(word)):
    #     if weight[1][j]!=0:
    #         print(word[j], weight[1][j])

    return weight


def Kmeans(weight, clusters, correct):
    mykms = KMeans(n_clusters=clusters)
    y = mykms.fit_predict(weight)
    result = []

    for i in range(0, clusters):
        label_i = []
        gp = 0
        jy = 0
        xz = 0
        ty = 0
        for j in range(0, len(y)):
            if y[j] == i:
                label_i.append(labels[j])
                type = labels[j][0:2]
                if (type == 'gp'):
                    gp += 1
                elif (type == 'jy'):
                    jy += 1
                elif (type == 'xz'):
                    xz += 1
                elif (type == 'ty'):
                    ty += 1
        max = jy
        type = '教育'
        if (gp > jy):
            max = gp
            type = '股票'
        if (max < xz):
            max = xz
            type = '星座'
        if (max < ty):
            max = ty
            type = '体育'
        correct[0] += max
        result.append('类别' + '(' + type + ')' + ':' + str(label_i))
    return result


def output(result, outputDir, clusters):
    outputFile = 'out'
    type = '.txt'
    count = 0
    while (os.path.exists(outputDir + outputFile + type)):
        count += 1
        outputFile = 'out' + str(count)
    doc = open(outputDir + outputFile + type, 'w')
    for i in range(0, clusters):
        print(result[i], file=doc)
    print('本次分类总样本数目为:' + str(size) + ' 其中正确分类数目为:' + str(correct[0]) + ' 正确率为：' + str(correct[0] / size), file=doc)
    doc.close()


texts = buildSW()
corpus = buildWB(texts)
weight = countIdf(corpus)
clusters = 4
correct = [0]  # 正确量
result = Kmeans(weight, clusters, correct)
output(result, outputDir, clusters)
print('finish')