import argparse
import os
import sys

import numpy as np
import torch
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from transformers import BertTokenizer, BertModel

from dataloader.dataloader import unshuffle_loader
from models.Transformers import SCCLBert
from training import SCCLvTrainer
from utils.kmeans import get_kmeans_centers
from utils.logger import set_global_random_seed, setup_path
from utils.metric import Confusion

import matplotlib.pyplot as plt

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# torch.cuda.set_device(2)


def evaluate_embedding(self):
    dataloader = unshuffle_loader(self.args)
    self.model.eval()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            text, label = batch['text'], batch['label']
            feat = self.get_batch_token(text)
            embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

            model_prob = self.model.get_cluster_prob(embeddings)
            if i == 0:
                all_labels = label
                all_embeddings = embeddings.detach()
                all_prob = model_prob
            else:
                all_labels = torch.cat((all_labels, label), dim=0)
                all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                all_prob = torch.cat((all_prob, model_prob), dim=0)

    # Initialize confusion matrices
    confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)

    all_pred = all_prob.max(1)[1]
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(self.args.num_classes)
    acc_model = confusion_model.acc()
    embeddings = all_embeddings.cpu().numpy()
    np.save("./embeddings.npy", embeddings)
    kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)

    kmeans.fit(embeddings)

    #
    # print(f"轮廓系数 by sklearn: {silhouette_score(embeddings, kmeans.labels_.astype(np.int))}")
    # print(f"方差比 by sklearn: {calinski_harabasz_score(embeddings, kmeans.labels_.astype(np.int))}")
    # print(f"db_score by sklearn: {davies_bouldin_score(embeddings, kmeans.labels_.astype(np.int))}")

    # dbscan = DBSCAN(eps=1, min_samples=10)
    # dbscan.fit(embeddings)

    print(f"轮廓系数 by sklearn: {silhouette_score(embeddings, kmeans.labels_.astype(np.int))}")
    print(f"方差比 by sklearn: {calinski_harabasz_score(embeddings, kmeans.labels_.astype(np.int))}")
    print(f"db_score by sklearn: {davies_bouldin_score(embeddings, kmeans.labels_.astype(np.int))}")

    # Initialize confusion matrices
    confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)

    all_pred = all_prob.max(1)[1]
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(self.args.num_classes)
    acc_model = confusion_model.acc()
    embeddings = all_embeddings.cpu().numpy()
    pred_labels = torch.tensor(kmeans.labels_.astype(np.int))

    # clustering accuracy
    confusion.add(pred_labels, all_labels)
    confusion.optimal_assignment(self.args.num_classes)
    acc = confusion.acc()

    ressave = {"acc": acc, "acc_model": acc_model}
    ressave.update(confusion.clusterscores())

    print('[Representation] Clustering scores:', confusion.clusterscores())
    print('[Representation] ACC: {:.3f}'.format(acc))
    print('[Model] Clustering scores:', confusion_model.clusterscores())
    print('[Model] ACC: {:.3f}'.format(acc_model))

    # np.save("./emb.npy", embeddings)
    # np.save("./kmeans.npy", labels)

    tsne = TSNE(n_components=2)
    data = tsne.fit_transform(embeddings)  # 进行数据降维,并返回结果
    labels = kmeans.labels_.astype(np.int)
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

    d = data_tsne[data_tsne.index == 3]
    plt.scatter(d[0], d[1], c='red', marker='o')

    d = data_tsne[data_tsne.index == 4]
    plt.scatter(d[0], d[1], c='black', marker='o')
    # plt.savefig('./trans_show.jpg')

    # pred_labels = torch.tensor(kmeans.labels_.astype(np.int))




def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[-1],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=100, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')

    parser.add_argument('--bert', type=str, default='guwen', help="")
    parser.add_argument('--use_pretrain', type=str, default='BERT', choices=["BERT", "SBERT", "PAIRSUPCON"])

    # Dataset
    parser.add_argument('--datapath', type=str, default='./data/')
    parser.add_argument('--dataname', type=str, default='origin', help="")
    parser.add_argument('--num_classes', type=int, default=9, help="")
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--augmentation_1', type=str, default='text1')
    parser.add_argument('--augmentation_2', type=str, default='text2')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=1000)
    # contrastive learning
    parser.add_argument('--objective', type=str, default='SCCL')
    parser.add_argument('--augtype', type=str, default='virtual', choices=['virtual', 'explicit'])
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=1, help="")

    # Clustering
    parser.add_argument('--alpha', type=float, default=1.0)

    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args

def evaluate(self):
    dataloader = unshuffle_loader(self.args)
    print('---- {} evaluation batches ----'.format(len(dataloader)))

    self.model.eval()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            text, label = batch['text'], batch['label']
            feat = self.get_batch_token(text)
            embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

            model_prob = self.model.get_cluster_prob(embeddings)
            if i == 0:
                all_labels = label
                all_embeddings = embeddings.detach()
                all_prob = model_prob
            else:
                all_labels = torch.cat((all_labels, label), dim=0)
                all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                all_prob = torch.cat((all_prob, model_prob), dim=0)

    # Initialize confusion matrices
    confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)

    all_pred = all_prob.max(1)[1]
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(self.args.num_classes)
    acc_model = confusion_model.acc()

    kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
    embeddings = all_embeddings.cpu().numpy()
    kmeans.fit(embeddings)
    pred_labels = torch.tensor(kmeans.labels_.astype(np.int))

    # clustering accuracy
    confusion.add(pred_labels, all_labels)
    confusion.optimal_assignment(self.args.num_classes)
    acc = confusion.acc()

    ressave = {"acc": acc, "acc_model": acc_model}
    ressave.update(confusion.clusterscores())

    # np.save(self.args.resPath + 'acc_{}.npy'.format(step), ressave)
    # np.save(self.args.resPath + 'scores_{}.npy'.format(step), confusion.clusterscores())
    # np.save(self.args.resPath + 'mscores_{}.npy'.format(step), confusion_model.clusterscores())
    # np.save(self.args.resPath + 'mpredlabels_{}.npy'.format(step), all_pred.cpu().numpy())
    # np.save(self.args.resPath + 'predlabels_{}.npy'.format(step), pred_labels.cpu().numpy())
    # np.save(self.args.resPath + 'embeddings_{}.npy'.format(step), embeddings)
    # np.save(self.args.resPath + 'labels_{}.npy'.format(step), all_labels.cpu())

    print('[Representation] Clustering scores:', confusion.clusterscores())
    print('[Representation] ACC: {:.3f}'.format(acc))
    print('[Model] Clustering scores:', confusion_model.clusterscores())
    print('[Model] ACC: {:.3f}'.format(acc_model))
    return None



if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)
    torch.cuda.set_device(args.gpuid[0])
    tokenizer = BertTokenizer.from_pretrained("chinese_rbt3_pytorch")
    bert = BertModel.from_pretrained("chinese_rbt3_pytorch")
    # tokenizer = BertTokenizer.from_pretrained("guwen")
    # bert = BertModel.from_pretrained("guwen")
    # tokenizer = BertTokenizer.from_pretrained("dist")
    # bert = BertModel.from_pretrained("dist")
    # cluster_centers = get_kmeans_centers(bert, tokenizer, unshuffle_loader(args), args.num_classes, args.max_length)
    # cluster_centers = np.load("./center.npz.npy")
    cluster_centers = torch.rand(args.num_classes, 768)
    model = SCCLBert(bert, tokenizer, cluster_centers=cluster_centers, alpha=args.alpha)
    model.load_state_dict(torch.load("simbertAug/model_2000.pth"))
    model = model.cuda()

    trainer = SCCLvTrainer(model, tokenizer, None, None, args)

    evaluate_embedding(trainer)
    # evaluate(trainer)

    # res = test("昨日(10月10日），无锡货车超载导致的高架侧翻事故，引发了外界对于桥梁安全问题的关注。")
    # print(res)
