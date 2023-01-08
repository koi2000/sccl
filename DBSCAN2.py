from _datetime import datetime
import json
import time

import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import jieba
from gensim.models import word2vec
from gensim.models import Word2Vec
import pandas as pd

model = Word2Vec.load("./checkpoint2/baidu.model")

from numpy import shape


# 计算两个向量之间的欧式距离
def calDist(X1, X2):
    sum = 0
    for x1, x2 in zip(X1, X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5


def wmdistance(text1, text2):
    distance = model.wv.wmdistance(text1, text2)
    return distance


# 获取一个点的ε-邻域（记录的是索引）
def getNeibor(data, dataSet, e):
    res = []
    for i in range(shape(dataSet)[0]):
        if wmdistance(data, dataSet[i]) < e:
            res.append(i)
    return res


# 密度聚类算法
def MyDBSCAN(dataSet, e, minPts):
    coreObjs = {}  # 初始化核心对象集合
    C = {}
    n = shape(dataSet)[0]
    # 找出所有核心对象，key是核心对象的index，value是ε-邻域中对象的index
    for i in range(n):
        neibor = getNeibor(dataSet[i], dataSet, e)
        if len(neibor) >= minPts:
            coreObjs[i] = neibor
    oldCoreObjs = coreObjs.copy()
    k = 0  # 初始化聚类簇数
    notAccess = list(range(n))  # 初始化未访问样本集合（索引）
    while len(coreObjs) > 0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = coreObjs.keys()
        # 随机选取一个核心对象
        randNum = random.randint(0, len(cores) - 1)
        cores = list(cores)
        core = cores[randNum]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue) > 0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys():
                delte = [val for val in oldCoreObjs[q] if val in notAccess]  # Δ = N(q)∩Γ
                queue.extend(delte)  # 将Δ中的样本加入队列Q
                notAccess = [val for val in notAccess if val not in delte]  # Γ = Γ\Δ
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    return C


def loadDataSet(filename):
    dataSet = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = map(float, curLine)
        # python2.x map函数返回list
        # dataSet.append(fltLine)
        # python3.x map函数返回迭代器
        dataSet.append(list(fltLine))
    return dataSet


def draw(C, dataSet):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for j in range(len(datas)):
            X.append(dataSet[datas[j]][0])
            Y.append(dataSet[datas[j]][1])
        plt.scatter(X, Y, marker='o', color=color[i % len(color)], label=i)
    plt.legend(loc='upper right')
    plt.show()


def main():
    # dataSet = loadDataSet("simpleDataSet.txt")
    df = pd.read_csv("./data/middleorigin.csv")
    # df = pd.read_csv("./data/smallorigin.csv")
    dataSet = list(df["text"].values)
    print(dataSet)

    start = datetime.now()
    C = MyDBSCAN(dataSet, 0.001, 3)
    # draw(C, dataSet)
    with open('dbscan3.json', 'w') as f:
        json.dump(C, f)
    end = datetime.now()
    print('用时：{:.4f}s'.format((end - start).seconds))

if __name__ == '__main__':
    main()
