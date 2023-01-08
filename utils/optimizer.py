"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch
from transformers import get_linear_schedule_with_warmup, BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer

BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased',
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
}


def get_optimizer(model, args):

    optimizer = torch.optim.Adam([
        {'params':model.bert.parameters()},
        {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale},
        {'params':model.cluster_centers, 'lr': args.lr*args.lr_scale}
    ], lr=args.lr)

    print(optimizer)
    return optimizer


def get_bert(args):
    # tokenizer = BertTokenizer.from_pretrained(args.bert)
    # # bert_model = BertModel.from_pretrained("bert-base-uncased")
    # print("bert层模型创建完成")
    # # bert_model = torch.load('bert_90.pth')
    # bert_model = BertModel.from_pretrained(args.bert)
    # return bert_model, tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    model = BertModel.from_pretrained(args.bert)
    return model,tokenizer
    # if args.use_pretrain == "SBERT":
    #     bert_model = get_sbert(args)
    #     tokenizer = bert_model[0].tokenizer
    #     model = bert_model[0].auto_model
    #     print("..... loading Sentence-BERT !!!")
    # else:
    #     # config = AutoConfig.from_pretrained(BERT_CLASS[args.bert],mirror='tuna')
    #     # model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
    #     # tokenizer = AutoTokenizer.from_pretrained("BERT_CLASS[args.bert]")
    #     config = AutoConfig.from_pretrained("guwen")
    #     model = AutoModel.from_pretrained("guwen", config=config)
    #     tokenizer = AutoTokenizer.from_pretrained("guwen")
    #     print("..... loading plain BERT !!!")
    # #
    # return model, tokenizer


def get_sbert(args):
    # sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    sbert = SentenceTransformer(args.bert)
    return sbert








