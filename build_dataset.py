import os
import shutil
import random


if __name__ == "__main__":
    data_path_old = r"D:\MyFiles\研究生课程\语义计算\大作业\数据\THUCNews"
    data_path_new = "data/THUCNews_custom"
    if not os.path.exists(data_path_new):
        os.makedirs(data_path_new)

    # classes = os.listdir(data_path_old)
    # print(classes)
    # classes = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
    # classes_nickname = ['ty', 'yl', 'jj', 'cp', 'fc', 'jy', 'ss', 'sz', 'xz', 'yx', 'sh', 'kj', 'gp', 'cj']

    classes = ['股票', '教育', '体育', '星座']
    classes_nickname = ['gp', 'jy', 'ty', 'xz']
    num_file_per_class = 200

    for i in range(len(classes)):
        class_path = os.path.join(data_path_old, classes[i])
        files = os.listdir(class_path)
        print("{}\t文件总数量：{}".format(class_path, len(files)))
        random_index = random.sample(range(0, len(files)), num_file_per_class)   # 随机选取
        # print(random_index)

        for j in range(num_file_per_class):
            # file_path_old = os.path.join(data_path_old, classes[i], files[j])     # 按顺序选取
            file_path_old = os.path.join(data_path_old, classes[i], files[random_index[j]])     # 随机选取
            # print(file_path_old)
            file_path_new = os.path.join(data_path_new, classes_nickname[i] + str(j) + ".txt")
            shutil.copyfile(file_path_old, file_path_new)

