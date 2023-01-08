import pandas as pd
import numpy as np
from nlpcda.tools.Simbert import Simbert
from tqdm import tqdm, trange
from nlpcda import Randomword


def randomword_augment(data_source, data_target, textcol="text"):
    train_data = pd.read_csv(data_source)

    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    augtxts1, augtxts2 = [], []
    for txt in tqdm(train_text):
        smw = Randomword(create_num=10, change_rate=0.3)
        rs1 = smw.replace(txt)

        # print('随机实体替换>>>>>>')
        # for s in rs1:
        #     print(s)

        if len(rs1) >= 2:
            augtxts1.append(str(rs1[0]))
            augtxts2.append(str(rs1[1]))
        elif len(rs1) >= 1:
            augtxts1.append(str(rs1[0]))
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

if __name__ == '__main__':
    randomword_augment("./org/origin.csv", "./randomword_origin.csv", textcol="text")