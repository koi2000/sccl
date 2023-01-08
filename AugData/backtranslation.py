import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm, trange
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelWithLMHead.from_pretrained("./opus-mt-en-zh")
tokenizer = AutoTokenizer.from_pretrained("./opus-mt-en-zh")

en2zh_translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)

# 中文翻译成英文
model = AutoModelWithLMHead.from_pretrained("opus-mt-zh-en/")
tokenizer = AutoTokenizer.from_pretrained("opus-mt-zh-en/")
zh2en_translation = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)



def randomword_augment(data_source, data_target, textcol="text"):
    train_data = pd.read_csv(data_source)

    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    augtxts1, augtxts2 = [], []
    for txt in tqdm(train_text):
        en = zh2en_translation(txt, max_length=128)[0]['translation_text']
        zh = en2zh_translation(en, max_length=128)[0]['translation_text']
        print(zh)
        augtxts1.append(zh)
        augtxts2.append(zh)

    train_data[textcol + "1"] = pd.Series(augtxts1)
    train_data[textcol + "2"] = pd.Series(augtxts2)
    train_data.to_csv(data_target, index=False)


if __name__ == '__main__':
    dir = './data'
    randomword_augment("./org/origin.csv", "./trans_origin.csv", textcol="text")
    # translate_batch(os.path.join(dir, 'insurance_train'), batch_num=30)