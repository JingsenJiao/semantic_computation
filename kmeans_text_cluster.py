# -*- coding: utf-8 -*-
import os
import re
import jieba
from sklearn import manifold
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from PIL import ImageFont
import matplotlib.pyplot as plt


inputDir = "data/THUCNews_custom"
outputDir = "data/output"  # 结果输出地址
processDir = "data/THUCNews_custom_processed"     # 切词和去停后的文件地址
figureDir = "data/figure"

if not os.path.exists(outputDir):
    os.makedirs(outputDir)
if not os.path.exists(processDir):
    os.makedirs(processDir)
if not os.path.exists(figureDir):
    os.makedirs(figureDir)

all_file = os.listdir(inputDir)  # 获取文件夹中所有文件名
labels = []  # 用以存储文档名称
corpus = []  # 语料库
# size = 200  # 测试集容量
size = len(all_file)


def buildSW():
    """停用词的建立"""
    # typetxt = open('data/stopwords/中文停用词_csdn.txt', encoding="utf-8")  # 停用词文档地址
    typetxt = open('data/stopwords/stopwords.txt', encoding="utf-8")  # 停用词文档地址
    texts = ['\u3000', '\n', ' ']  # 爬取的文本中未处理的特殊字符，u3000 是全角的空白符
    '''停用词库的建立'''
    for word in typetxt:
        word = word.strip()
        texts.append(word)
    typetxt.close()
    return texts


def buildWB(texts):
    """语料库的建立"""
    for i in range(0, len(all_file)):
        filename = all_file[i]
        filelabel = filename.split('.')[0]
        labels.append(filelabel)  # 名称列表
        file_add = os.path.join(inputDir, filename)  # 文档的路径
        doc = open(file_add, encoding='utf-8').read()
        data = jieba.cut(doc)  # 文本分词
        data_adj = ''
        delete_word = []
        for item in data:
            if item not in texts:  # 停用词过滤
                # data_adj += item + ' '
                # value=re.compile(r'^[0-9]+$')#去除数字
                value = re.compile(r'^[\u4e00-\u9fa5]{2,}$')  # 只匹配中文2字词以上
                if value.match(item):
                    data_adj += item + ' '
            else:
                delete_word.append(item)
        corpus.append(data_adj)  # 语料库建立完成
        processed_file = os.path.join(processDir, filename)
        processed_doc = open(processed_file, 'w', encoding='utf-8')
        print(data_adj, file=processed_doc)
        processed_doc.close()
    # print(corpus)
    return corpus


def countIdf(corpus):
    # 分词向量化
    vectorizer = CountVectorizer()      # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    word_vec = vectorizer.fit_transform(corpus)     # 将文本转为词频矩阵

    # 提取 TF-IDF 词向量
    transformer = TfidfTransformer()    # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(word_vec)     # 计算tf-idf
    tfidf_matrix = tfidf.toarray()      # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
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
    mykms = KMeans(n_clusters=clusters, max_iter=200)
    kmeans = mykms.fit(weight)
    y = mykms.fit_predict(weight)
    result = []

    # 打印出各个族的中心点
    print(kmeans.cluster_centers_)
    print(kmeans.cluster_centers_.shape)
    # for index, label in enumerate(kmeans.labels_, 1):
    #     print("index: {}, label: {}".format(index, label))

    # 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
    # k-means 的超参数 n_clusters 可以通过该值来评估
    print("inertia: {}".format(kmeans.inertia_))

    # 评价指标
    real_labels = ['股票']*200 + ['教育']*200 + ['体育']*200 + ['星座']*200
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(real_labels, kmeans.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(real_labels, kmeans.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(real_labels, kmeans.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(real_labels, kmeans.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(weight, kmeans.labels_, metric='euclidean'))

    # 绘制聚类后各簇的词云图
    text_classes = {0: '', 1: '', 2: '', 3: ''}
    for k in range(len(text_classes)):
        text_cls = []
        for file, res in zip(all_file, y):
            if res == k:
                text = open(os.path.join(processDir, file), encoding='utf-8')
                text_cls.append(text.read())
                text.close()
        text_join = ''.join(line for line in text_cls)
        text_classes[k] = text_join
        generate_wordclouds(text_join, os.path.join(figureDir, str(k) + '.png'))  # 绘制词云图

    # 绘制聚类后的散点图，不同的簇用不同颜色表示
    # markers = ['^', 'v', '<', '>', 's', 'o', '.', '*']
    # colors = ['r', 'g', 'b', 'm', 'k', 'y', 'g', 'r']
    markers = ['s', 'o', '*', '^']
    colors = ['r', 'g', 'b', 'm']
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(len(y)):
        plt.scatter(weight[i, 0], weight[i, 1], c=colors[y[i]], marker=markers[y[i]])
        plt.text(weight[i, 0], weight[i, 1] + 0.01, '%d' % y[i], ha='center', va='bottom', fontsize=7)
    for i in range(clusters):
        plt.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1], c='k', marker='x', linewidths=5)
    fig.savefig('data/figure/kmeans.png', transparent=False, dpi=100, bbox_inches="tight")

    # 统计结果
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
                if type == 'gp':
                    gp += 1
                elif type == 'jy':
                    jy += 1
                elif type == 'xz':
                    xz += 1
                elif type == 'ty':
                    ty += 1
        max = jy
        type = '教育'
        if gp > max:
            max = gp
            type = '股票'
        if xz > max:
            max = xz
            type = '星座'
        if ty > max:
            max = ty
            type = '体育'
        correct[0] += max
        result.append('类别' + '(' + type + ')' + ':' + str(label_i))
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
    print('本次分类总样本数目为:' + str(size) + ' 其中正确分类数目为:' + str(correct[0]) + ' 正确率为：' + str(correct[0] / size), file=doc)
    doc.close()


def generate_wordclouds(text, out_file):
    # 设置停用词
    stopwords = set(STOPWORDS)      # wordcloud 自带的停用词只有英文
    # stopwords.add(r"weapon")
    # stopwords.add(r"huanqiu")

    # 指定字体为中文
    font = r'C:\Windows\Fonts\msyh.ttc'
    words = WordCloud(font_path=font, background_color="white", stopwords=stopwords, width=1200, height=800, margin=2)

    # 生成词云图
    words.generate(text)
    words.to_file(out_file)


texts = buildSW()
corpus = buildWB(texts)
weight = countIdf(corpus)
clusters = 4
correct = [0]  # 正确量
result = Kmeans(weight, clusters, correct)
output(result, outputDir, clusters)

# 显示散点图
img = Image.open(os.path.join(figureDir, 'kmeans.png'))
plt.figure(figsize=(20, 16))
plt.imshow(img)
plt.axis('off')         # 关掉坐标轴为 off
plt.title('Cluster Result')
plt.show()

# 显示词云图
plt.figure(figsize=(20, 20))
for i in range(4):
    img = Image.open(os.path.join(figureDir, str(i)+'.png'))
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.axis('off')     # 关掉坐标轴为 off
plt.show()

print('finish')
