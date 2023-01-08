import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn import cluster
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


def evaluate_embedding(self):
    # 读取到数据
    dataloader = unshuffle_loader(self.args)
    self.model.eval()
    textList = []
    # 将所有的文本转为向量
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            text, label = batch['text'], batch['label']
            # 将所有的文本保存起来
            textList.append(text)
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
    kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
    embeddings = all_embeddings.cpu().numpy()
    kmeans.fit(embeddings)
    labels = kmeans.labels_.astype(np.int)
    label2text = {}
    idx = 0
    for label in labels:
        label = int(label)
        if label in label2text.keys():
            label2text[label].append(textList[idx])
        else:
            label2text[label] = []

    with open('label2text.json', 'w') as f:
        json.dump(label2text, f)
    np.save("./label2text.npy", label2text)
    np.save("./emb.npy", embeddings)
    np.save("./kmeans.npy", labels)


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[-1],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=100, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')

    parser.add_argument('--bert', type=str, default='distilbert', help="")
    parser.add_argument('--use_pretrain', type=str, default='BERT', choices=["BERT", "SBERT", "PAIRSUPCON"])

    # Dataset
    parser.add_argument('--datapath', type=str, default='./data/')
    parser.add_argument('--dataname', type=str, default='clusterdata', help="")
    parser.add_argument('--num_classes', type=int, default=8, help="")
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


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)
    torch.cuda.set_device(args.gpuid[0])
    tokenizer = BertTokenizer.from_pretrained("guwen")
    bert = BertModel.from_pretrained("guwen")
    # tokenizer = BertTokenizer.from_pretrained("dist")
    # bert = BertModel.from_pretrained("dist")
    # cluster_centers = get_kmeans_centers(bert, tokenizer, unshuffle_loader(args), args.num_classes, args.max_length)
    # cluster_centers = np.load("./center.npz.npy")
    cluster_centers = torch.rand(8, 768)
    model = SCCLBert(bert, tokenizer, cluster_centers=cluster_centers, alpha=args.alpha)
    model.load_state_dict(torch.load("checkpoint2/scclmodel_3000.pth"))
    model = model.cuda()

    trainer = SCCLvTrainer(model, tokenizer, None, None, args)

    evaluate_embedding(trainer)
