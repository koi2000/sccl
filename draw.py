import joblib
from sklearn import cluster
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kmeans = joblib.load("kmeans_3898.pkl")

# kmeans = cluster.py.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
# embeddings = all_embeddings.cpu().numpy()
# kmeans.fit(embeddings)

labels = kmeans.labels_.astype(np.int)
np.save("./kmeans.npy", labels)
tsne = TSNE(n_components=2)
data = tsne.fit_transform(embeddings)  # 进行数据降维,并返回结果

# 将原始数据中的索引设置成聚类得到的数据类别
data = pd.DataFrame(data, index=labels)
data_tsne = pd.DataFrame(tsne.embedding_, index=data.index)

# 根据类别分割数据后，画图
d = data_tsne[data_tsne.index == 0]  # 找出聚类类别为0的数据对应的降维结果
plt.scatter(d[0], d[1], c='lightgreen', marker='o')
d = data_tsne[data_tsne.index == 1]
plt.scatter(d[0], d[1], c='orange', marker='o')
d = data_tsne[data_tsne.index == 2]
plt.scatter(d[0], d[1], c='lightblue', marker='o')
plt.savefig('./res.jpg')