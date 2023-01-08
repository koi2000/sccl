import argparse
import json
import sys

import numpy as np

from dataloader.dataloader import unshuffle_loader


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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=1, help="")

    # Clustering
    parser.add_argument('--alpha', type=float, default=1.0)

    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args


dataloader = unshuffle_loader(get_args(sys.argv[1:]))
textList = []
# 将所有的文本转为向量
for i, batch in enumerate(dataloader):
    text, label = batch['text'], batch['label']
    # 将所有的文本保存起来
    textList.append(text)

label2text = np.load("./kmeans.npy", allow_pickle=True)

mp = {}
for i in range(len(label2text)):
    keys = int(label2text[i])
    if keys in mp.keys():
        mp[keys].append(textList[i])
    else:
        mp[keys] = [textList[i]]
    if i == 100000:
        break


with open('label2text2.json', 'w') as f:
    json.dump(mp, f)

print(label2text)
