# -*- coding: utf-8 -*-
# 随机选择若干类文本，然后根据轮廓系数计算最优聚类类别数，再调用聚类算法
import os
import re
import jieba
import shutil
import random
from sklearn import manifold
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from PIL import ImageFont
import numpy as np
import matplotlib.pyplot as plt


inputDir = "data/THUCNews_test"
outputDir = "data/output_test"  # 结果输出地址
processDir = "data/THUCNews_test_processed"     # 切词和去停后的文件地址
figureDir = "data/figure_test"

if os.path.exists(inputDir):
    for filename in os.listdir(inputDir):
        os.remove(os.path.join(inputDir, filename))  # 删除原有文件
else:
    os.makedirs(inputDir)

if os.path.exists(processDir):
    for filename in os.listdir(processDir):
        os.remove(os.path.join(processDir, filename))  # 删除原有文件
else:
    os.makedirs(processDir)

if not os.path.exists(outputDir):
    os.makedirs(outputDir)
if not os.path.exists(figureDir):
    os.makedirs(figureDir)

all_file = []
labels = []  # 用以存储文档名称
corpus = []  # 语料库
size = 0


def build_random_dataset(min_num_class=2, max_num_class=14, num_file_per_class=100):
    data_path_old = r"D:\MyFiles\研究生课程\语义计算\大作业\数据\THUCNews"
    data_path_new = inputDir

    # classes = os.listdir(data_path_old)
    # print(classes)
    classes = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
    # classes_nickname = ['ty', 'yl', 'jj', 'cp', 'fc', 'jy', 'ss', 'sz', 'xz', 'yx', 'sh', 'kj', 'gp', 'cj']
    num_class = random.randint(min_num_class, max_num_class)         # randint 是闭区间，最少取 2 类
    print("随机取 {} 类: ".format(num_class), end=' ')
    class_index = random.sample(range(0, len(classes)), num_class)        # range 是左闭右开
    class_index.sort()
    for i in class_index:
        print(classes[i], end=' ')
    print("\n每类取 {} 个文件".format(num_file_per_class))

    for i in class_index:
        class_path = os.path.join(data_path_old, classes[i])
        files = os.listdir(class_path)
        print("{}\t文件总数量：{}".format(classes[i], len(files)))
        random_index = random.sample(range(0, len(files)), num_file_per_class)   # 随机选取
        # print(random_index)

        for j in range(num_file_per_class):
            # file_path_old = os.path.join(data_path_old, classes[i], files[j])     # 按顺序选取
            file_path_old = os.path.join(data_path_old, classes[i], files[random_index[j]])     # 随机选取
            # print(file_path_old)
            file_path_new = os.path.join(data_path_new, classes[i] + "_" + str(j) + ".txt")
            shutil.copyfile(file_path_old, file_path_new)
    return num_class


def build_stopwords():
    """停用词的建立"""
    # typetxt = open('data/stopwords/中文停用词_csdn.txt', encoding="utf-8")  # 停用词文档地址
    typetxt = open('data/stopwords/stopwords.txt', encoding="utf-8")  # 停用词文档地址
    stopwords = ['\u3000', '\n', ' ']  # 特殊字符，u3000 是全角的空白符
    '''停用词库的建立'''
    for word in typetxt:
        word = word.strip()
        stopwords.append(word)
    typetxt.close()
    return stopwords


def build_corpus(stopwords):
    """语料库的建立"""
    for i in range(0, len(all_file)):
        filename = all_file[i]
        file_add = os.path.join(inputDir, filename)  # 文档的路径
        doc = open(file_add, encoding='utf-8')
        data = jieba.cut(doc.read())  # 文本分词
        doc.close()

        data_adj = ''
        # delete_word = []
        for item in data:
            if item not in stopwords:  # 停用词过滤
                # data_adj += item + ' '
                # value = re.compile(r'^[0-9]+$')  # 去除数字
                value = re.compile(r'^[\u4e00-\u9fa5]{2,}$')  # 只匹配中文2字词以上
                if value.match(item):
                    data_adj += item + ' '
            # else:
                # delete_word.append(item)
        corpus.append(data_adj)  # 将该文档内的词加入语料库，corpus的元素数量就是总文档数，其中每个元素为该文档内的所有词(词间以空格分隔)

        processed_file = os.path.join(processDir, filename)
        processed_doc = open(processed_file, 'w', encoding='utf-8')
        print(data_adj, file=processed_doc)     # 将处理后的文本写入文件
        processed_doc.close()
    # print(corpus)
    # print(len(corpus))
    return corpus


def count_tfidf(corpus):
    # 分词向量化
    vectorizer = CountVectorizer()                  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    word_vec = vectorizer.fit_transform(corpus)     # 将文本转为词频矩阵

    # 提取 TF-IDF 词向量
    transformer = TfidfTransformer()                # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(word_vec)     # 计算tf-idf
    tfidf_matrix = tfidf.toarray()                  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    print("TF-IDF 矩阵的维数：{}".format(tfidf_matrix.shape))

    # tsne 降维
    # tf-idf 矩阵的行数为总文档数量，列数为所有文档分词去停后所有词的数量，维数很高，而且无法在平面图上画出图形，因此需要降维
    # 降维后，可以把聚类准确率从 0.7 提高到 0.9
    tsne = manifold.TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0,
                         learning_rate=200.0, n_iter=1000, init="pca", random_state=0)
    tsne_tfidf_matrix = tsne.fit_transform(tfidf_matrix)
    print("降维后 TF-IDF 矩阵的维数：{}".format(tsne_tfidf_matrix.shape))

    # word = vectorizer.get_feature_names()         # 获取词袋模型中的所有词
    # for j in range(len(word)):
    #     if weight[1][j] != 0:
    #         print(word[j], weight[1][j])

    return tsne_tfidf_matrix


def Kmeans(weight, clusters, correct):
    mykms = KMeans(n_clusters=clusters, max_iter=300)
    kmeans = mykms.fit(weight)
    y = mykms.fit_predict(weight)
    result = []

    # 打印出各个簇的中心点
    # print(kmeans.cluster_centers_)
    # print(kmeans.cluster_centers_.shape)
    # for index, label in enumerate(kmeans.labels_, 1):
    #     print("index: {}, label: {}".format(index, label))

    # 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
    # k-means 的超参数 n_clusters 可以通过该值来评估
    print("inertia: {}".format(kmeans.inertia_))

    # # 评价指标
    # real_labels = ['股票']*200 + ['教育']*200 + ['体育']*200 + ['星座']*200
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(real_labels, kmeans.labels_))
    # print("Completeness: %0.3f" % metrics.completeness_score(real_labels, kmeans.labels_))
    # print("V-measure: %0.3f" % metrics.v_measure_score(real_labels, kmeans.labels_))
    # print("Adjusted Rand-Index: %.3f"
    #       % metrics.adjusted_rand_score(real_labels, kmeans.labels_))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(weight, kmeans.labels_, metric='euclidean'))

    # 绘制聚类后的散点图，不同的簇用不同颜色表示
    markers = ['^', 'v', '<', '>', 's', 'o', '.', '*', 'x', '+', 'p', 'd', '1', '2']
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k', 'gray', 'brown', 'orangered', 'lime', 'deepskyblue', 'royalblue', 'deeppink']
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(len(y)):
        plt.scatter(weight[i, 0], weight[i, 1], c=colors[y[i]], marker=markers[y[i]])
        # plt.text(weight[i, 0], weight[i, 1] + 0.01, '%d' % y[i], ha='center', va='bottom', fontsize=7)
    # for i in range(clusters):
    #     plt.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1], c='k', marker='x', linewidths=5)
    fig.show()
    # fig.savefig('data/figure/kmeans.png', transparent=False, dpi=100, bbox_inches="tight")

    for i in range(0, len(all_file)):
        filename = all_file[i]
        filelabel = filename.split('.')[0]
        labels.append(filelabel)  # 文件名称列表

    # 统计结果
    for i in range(0, clusters):
        label_i = []
        for j in range(0, len(y)):
            if y[j] == i:
                label_i.append(labels[j])
        result.append('类别' + str(i) + ': ' + str(label_i) + '\n')
    return result


def output(result, outputDir, clusters):
    outputFile = 'out'
    type = '.txt'
    count = 0
    while os.path.exists(os.path.join(outputDir, outputFile + type)):
        count += 1
        outputFile = 'out' + str(count)
    doc = open(os.path.join(outputDir, outputFile + type), 'w', encoding='utf-8')
    for i in range(0, clusters):
        print(result[i], file=doc)
    doc.close()


# 若类别个数无法提前预知，采用轮廓系数计算最优的类别个数
def eval_kmeans(tfidf_matrix, start=2, end=14):
    # 初始化变量
    scores = []
    values = np.arange(start, end+1)

    # 迭代计算不同的轮廓系数值
    for num_clusters in values:
        # 训练kMeans模型
        km_cluster = KMeans(n_clusters=num_clusters, max_iter=200, n_init=40, init='k-means++')
        km_cluster.fit(tfidf_matrix)
        score = metrics.silhouette_score(tfidf_matrix, km_cluster.labels_,
                                         metric='euclidean', sample_size=len(tfidf_matrix))

        print("\nNumber of clusters =", num_clusters)
        print("Silhouette score =", score)
        scores.append(score)

    # 输出不同类别个数对应的轮廓系数值
    fig = plt.figure(figsize=(9, 4))
    plt.bar(values, scores, width=0.7, color='b', align='center')
    plt.ylim(0.0, 1.0)
    for a, b in zip(values, scores):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    fig.show()
    # fig.savefig('data/figure/silhouette.png', transparent=False, dpi=600, bbox_inches="tight")

    # Extract best score and optimal number of clusters
    num_clusters = np.argmax(scores) + values[0]
    return num_clusters


if __name__ == "__main__":
    num_class = build_random_dataset(min_num_class=3, max_num_class=5, num_file_per_class=100)
    all_file = os.listdir(inputDir)
    size = len(all_file)

    stopwords = build_stopwords()
    corpus = build_corpus(stopwords)
    weight = count_tfidf(corpus)

    clusters = eval_kmeans(weight, start=2, end=14)
    print("\n真实的类别数目为: ", num_class)
    print("计算出最优类别数目为: ", clusters)

    correct = [0]  # 正确量
    result = Kmeans(weight, clusters, correct)
    output(result, outputDir, clusters)

    # 显示聚类前散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(weight[:, 0], weight[:, 1], c='black', marker='o')
    plt.show()

    # # 显示聚类后散点图
    # img = Image.open(os.path.join(figureDir, 'kmeans.png'))
    # plt.figure(figsize=(20, 20))
    # plt.imshow(img)
    # plt.axis('off')         # 关掉坐标轴为 off
    # plt.title('Cluster Result')
    # plt.show()

    print('finish')
