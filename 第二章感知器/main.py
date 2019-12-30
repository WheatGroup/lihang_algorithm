# 针对minst数据集 进行感知器二分类的计算
# 加载minst文件
import pandas as pd
import numpy as np
import cv2
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):
    def __init__(self):
        # 学习率
        self.learning_step = 0.00001
        # 迭代的次数
        self.max_iteration = 5000
        # 最大连续正确
        self.max_correct = 3000


    # 利用opencv获取图像hog特征
    def get_hog_features(self, trainset):
        features = []

        hog = cv2.HOGDescriptor('../hog.xml')

        for img in trainset:
            img = np.reshape(img, (28, 28))
            cv_img = img.astype(np.uint8)

            hog_feature = hog.compute(cv_img)
            # hog_feature = np.transpose(hog_feature)
            features.append(hog_feature)

        features = np.array(features)
        features = np.reshape(features, (-1, 324))

        return features


    def train(self, feature, train_labels):
        # 获取参数
        feature_size = len(feature[0])
        self.w = np.zeros(feature_size)
        self.b = 0.0
        # 统计连续分类正确的次数  次数大于某值之后 认为 模型已经准确了
        correct_count = 0
        # 迭代的次数
        time = 0
        index = 10
        while time < self.max_iteration:
            # index = np.random.randint(0, feature_size)
            index += 10
            y = 2 * labels[index] - 1
            x = features[index]
            wx = self.w @ x
            b = self.b
            if y * (wx + b) <= 0:
                self.w += self.learning_step * (x * y)
                self.b += self.learning_step * y
            else:
                correct_count += 1
                if correct_count >= self.max_correct:
                    break


    def _predict(self, x_narray):
        wx = self.w @ x_narray
        return int((wx + self.b) > 0)

    def predict(self, features):
        labels = []
        for feature in features:
            labels.append(self._predict(feature))

        return labels


if __name__ == '__main__':
    p = Perceptron()
    print('read minist dataset')
    time_1 = time.time()

    raw_data = pd.read_csv('train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]
    # 对图像特征进行hog的特征提取
    # features = p.get_hog_features(imgs)
    features = imgs
    # 对特征数据集进行分割
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print('read data cost: ', time_2 - time_1, ' second', '\n')

    print('Start training')
    # 接着就需要用梯度下降 求解w b的数值
    p.train(train_features, train_labels)
    labels = p.predict(test_features)
    print(labels)







