import jieba
from gensim.models import word2vec
import gensim
import logging


# 此函数作用是对初始语料进行分词处理后，作为训练模型的语料
def cut_txt(old_file):
    global cut_file  # 分词之后保存的文件名
    cut_file = old_file + '_cut.txt'
    fi = open(old_file, 'r', encoding='utf-8')
    # try:
    #     fi = open(old_file, 'r', encoding='utf-8')
    # except BaseException as e:  # 因BaseException是所有错误的基类，用它可以获得所有错误类型
    #     print(Exception, ":", e)  # 追踪错误详细信息

    text = fi.read()  # 获取文本内容
    new_text = jieba.cut(text, cut_all=False)  # 精确模式
    str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')  # 去掉标点符号
    fo = open(cut_file, 'w', encoding='utf-8')
    fo.write(str_out)


def model_train(train_file_name, save_model_file):  # model_file_name为训练语料的路径,save_model为保存模型名
    # model = gensim.models.Word2Vec(sentences, sg=1, size=100, window=5, min_count=2, negative=3, sample=0.001, hs=1,
    #                                workers=4)
    # 该步骤也可分解为以下三步（但没必要）：
    # model=gensim.model.Word2Vec() 建立一个空的模型对象
    # model.build_vocab(sentences) 遍历一次语料库建立词典
    # model.train(sentences) 第二次遍历语料库建立神经网络模型

    # sg=1是skip—gram算法，对低频词敏感，默认sg=0为CBOW算法
    # size是神经网络层数，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
    # window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）
    # min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。
    # negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3,
    # negative: 如果>0,则会采用negativesamping，用于设置多少个noise words
    # hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
    # workers是线程数，此参数只有在安装了Cpython后才有效，否则只能使用单核

    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = gensim.models.Word2Vec(sentences, vector_size=200, window=10, min_count=1, workers=4,
                                   negative=3)  # 训练skip-gram模型; 默认window=5

    # model.build_vocab(sentences)
    # model.train(sentences)
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_file + ".bin", binary=True)  # 以二进制类型保存模型以便重用


if __name__ == '__main__':
    model_train("./data/txtfile.txt_cut.txt", "./checkpoint2/baidu.model")
    # cut_txt('./data/txtfile.txt')
