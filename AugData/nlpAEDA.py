import random

import pandas as pd
import numpy as np
from nlpcda.tools.Simbert import Simbert
from tqdm import tqdm, trange
from nlpcda import Randomword

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.3


def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(punc_ratio * len(words) + 1))
	qs = random.sample(range(0, len(words)), q)

	for j, word in enumerate(words):
		if j in qs:
			new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line

def randomword_augment(data_source, data_target, textcol="text"):
    train_data = pd.read_csv(data_source)

    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    augtxts1, augtxts2 = [], []
    for txt in tqdm(train_text):
        txt1 = insert_punctuation_marks(txt)
        txt2 = insert_punctuation_marks(txt)
        augtxts1.append(txt1)
        augtxts2.append(txt2)

    train_data[textcol + "1"] = pd.Series(augtxts1)
    train_data[textcol + "2"] = pd.Series(augtxts2)
    train_data.to_csv(data_target, index=False)

    for o, a1, a2 in zip(train_text[:5], augtxts1[:5], augtxts2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmentation1: \n", a1)
        print("-----Augmentation2: \n", a2)

if __name__ == '__main__':
    randomword_augment("./org/origin.csv", "./AEDA_origin.csv", textcol="text")