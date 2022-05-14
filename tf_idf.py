# encoding=utf-8

# https://blog.csdn.net/liuxuejiang158blog/article/details/31360765
# https://github.com/fxsjy/jieba
# https://scikit-learn.org/stable/modules/feature_extraction.html

import jieba
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

if __name__ == "__main__":
    # jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
    # strs = ["我来到北京清华大学", "乒乓球拍卖完了", "中国科学技术大学"]
    # for str in strs:
    #     seg_list = jieba.cut(str, use_paddle=True)  # 使用paddle模式
    #     print("Paddle Mode: " + '/'.join(list(seg_list)))

    seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    print("Full Mode: " + "/ ".join(seg_list))  # 全模式

    seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

    seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
    print(", ".join(seg_list))

    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    print(", ".join(seg_list))

    # jieba.lcut()直接返回列表
    seg_list = jieba.lcut("他来到了网易杭研大厦")  # 默认是精确模式
    print(seg_list)
    print(" ".join(seg_list))

    print("-" * 60)
    # corpus = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
    #           "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
    #           "小明 硕士 毕业 于 中国 科学院",  # 第三类文本的切词结果
    #           "我 爱 北京 天安门"]  # 第四类文本的切词结果
    strs = ["我来到北京清华大学", "他来到了网易杭研大厦", "小明硕士毕业于中国科学院", "我爱北京天安门"]
    corpus = []     # 分词之后的列表
    for str in strs:
        # corpus.append(" ".join(jieba.lcut(str)))
        corpus.append(" ".join(jieba.cut(str)))
    print(corpus)

    vectorizer = CountVectorizer()                  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()                # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))           # 外层fit_transform是计算tf-idf，内层fit_transform是将文本转为词频矩阵

    # word = vectorizer.get_feature_names()         # 获取词袋模型中的所有词语，get_feature_names()已过时
    word = vectorizer.get_feature_names_out()       # 获取词袋模型中的所有词语
    weight = tfidf.toarray()            # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    for i in range(len(weight)):        # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            print(word[j], weight[i][j])
