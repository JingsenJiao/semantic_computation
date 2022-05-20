# https://www.heywhale.com/mw/project/5f5dc9afae300e0046fdd488
# https://scikit-learn.org/stable/modules/clustering.html?highlight=silhouette_score#silhouette-coefficient
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import kmeans_text_cluster


# 若类别个数无法提前预知，采用轮廓系数计算最优的类别个数
def eval_kmeans(tfidf_matrix):
    # 初始化变量
    scores = []
    values = np.arange(2, 13)

    # 迭代计算不同的轮廓系数值
    for num_clusters in values:
        # 训练kMeans模型
        km_cluster = KMeans(n_clusters=num_clusters, max_iter=100, n_init=40, init='k-means++')
        km_cluster.fit(tfidf_matrix)
        score = metrics.silhouette_score(tfidf_matrix, km_cluster.labels_,
                                         metric='euclidean', sample_size=len(tfidf_matrix))

        print("\nNumber of clusters =", num_clusters)
        print("Silhouette score =", score)
        scores.append(score)

    # 输出不同类别个数对应的轮廓系数值
    fig = plt.figure(figsize=(9, 4))
    plt.bar(values, scores, width=0.7, color='b', align='center')
    plt.ylim(0.2, 0.7)
    for a, b in zip(values, scores):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    fig.show()
    fig.savefig('data/figure/silhouette.png', transparent=False, dpi=600, bbox_inches="tight")

    # Extract best score and optimal number of clusters
    num_clusters = np.argmax(scores) + values[0]
    print('\nOptimal number of clusters =', num_clusters)


if __name__ == "__main__":
    stopwords = kmeans_text_cluster.build_stopwords()
    corpus = kmeans_text_cluster.build_corpus(stopwords)
    weight = kmeans_text_cluster.count_tfidf(corpus)

    eval_kmeans(weight)
