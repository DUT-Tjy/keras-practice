from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def word2vec_weight(glove_path, train_sentence, test_sentence):
    """将GloVe词向量转换为Word2Vec格式"""
    # GloVe格式的输入文件
    glove_file = datapath("./glove.txt")
    # Word2Vec格式的输出文件
    word2vec_file = get_tmpfile("./word2vec.txt")

    # 进行格式转换
    glove2word2vec(glove_file, word2vec_file)

    # 转换后的文件
    model = KeyedVectors.load_word2vec_format(word2vec_file)

    all_sentence = train_sentence + test_sentence
    # 分词后的句子向量化表示，将文本转换为序列
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentence)

    # 转换为词向量矩阵
    vocab_size = len(tokenizer.word_index)
    # 计算未出现词数据个数
    error_count = 0
    glove_embedding_metrix = np.zeros((vocab_size + 1, 256))
    for word, i in tqdm(tokenizer.word_index.items()):
        if word in model:
            glove_embedding_metrix[i] = model.wv[word]
        else:
            error_count += 1
    print("embedding metrix构建完成，共有未知词"+str(error_count)+"个")

    # train/test数据进行padding处理
    train_padding = tokenizer.texts_to_sequences(train_sentence)
    test_padding = tokenizer.texts_to_sequences(test_sentence)
    # maxlen根据具体需求进行更改
    train_padding = pad_sequences(train_padding, maxlen=30)
    test_padding = pad_sequences(test_padding, maxlen=30)

    return train_padding, test_padding, glove_embedding_metrix, vocab_size
