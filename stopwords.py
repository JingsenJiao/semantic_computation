# 汇总停用词
import os


if __name__ == "__main__":
    file_dir = "data/stopwords"
    filenames = ["baidu_stopwords.txt", "cn_stopwords.txt", "hit_stopwords.txt", "scu_stopwords.txt"]
    output_filename = "stopwords.txt"
    # stopwords = {'\u3000', '\n', ' '}
    stopwords = set()

    for filename in filenames:
        file_path = os.path.join(file_dir, filename)
        typetxt = open(file_path, encoding="utf-8")
        for word in typetxt:
            word = word.strip()
            stopwords.add(word)
        typetxt.close()

    stopwords = list(stopwords)
    stopwords.sort()
    print("停用词总数量：" + str(len(stopwords)))

    output_file_path = os.path.join(file_dir, output_filename)
    doc = open(output_file_path, 'w', encoding='utf-8')
    for word in stopwords:
        # s = word.replace("'", '') + '\n'  # 去除单引号，每行末尾追加换行符
        s = word + '\n'
        doc.write(s)
    doc.close()

