import pandas as pd
import numpy as np
from nlpcda.tools.Simbert import Simbert
from tqdm import tqdm, trange

config = {
    'model_path': 'chinese_simbert_L-12_H-768_A-12',
    'CUDA_VISIBLE_DEVICES': '0',
    'max_len': 128,
    'seed': 1
}
simbert = Simbert(config=config)
sent = '把我的一个亿存银行安全吗'
print(sent)
synonyms = simbert.replace(sent=sent, create_num=2)
print(synonyms[0])
res = synonyms[1][0]
res = res.replace("万","亿")
print(res)

def simbert_augment(data_source, data_target, textcol="text"):

    train_data = pd.read_csv(data_source)

    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    augtxts1, augtxts2 = [], []
    for txt in tqdm(train_text):
        synonyms = simbert.replace(sent=txt, create_num=2)
        if len(synonyms) >= 2:
            augtxts1.append(str(synonyms[0][0]))
            augtxts2.append(str(synonyms[1][0]))
        elif len(synonyms) >= 1:
            augtxts1.append(str(synonyms[0][0]))
            augtxts2.append(txt)
        else:
            augtxts1.append(txt)
            augtxts2.append(txt)

    train_data[textcol + "1"] = pd.Series(augtxts1)
    train_data[textcol + "2"] = pd.Series(augtxts2)
    train_data.to_csv(data_target, index=False)

    for o, a1, a2 in zip(train_text[:5], augtxts1[:5], augtxts2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmentation1: \n", a1)
        print("-----Augmentation2: \n", a2)


# if __name__ == '__main__':
#     simbert_augment("./org/origin.csv", "./aug_origin.csv", textcol="text")
