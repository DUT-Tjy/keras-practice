from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm

"""输入为分词之后的语料"""
"""preprocessing.py进行jieba分词处理"""


def word2vec_weight(train_sentence, test_sentence):
    model_word2vec = Word2Vec(min_count=1, window=5, size=256, workers=4, batch_words=1000)
    # vocabulary
    all_sentence = train_sentence + test_sentence
    model_word2vec.build_vocab(all_sentence, progress_per=2000)
    model_word2vec.train(all_sentence,
                         total_examples=model_word2vec.corpus_count,
                         epochs=5,
                         compute_loss=True,
                         report_delay=60*10)
    print("Word2Vec模型训练完成！")

    # 分词后的句子向量化表示，将文本转换为序列
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentence)

    # 转换为词向量矩阵
    vocab_size = len(tokenizer.word_index)
    # 计算未出现词数据个数
    error_count = 0
    word2vec_embedding_metrix = np.zeros((vocab_size + 1, 256))
    for word, i in tqdm(tokenizer.word_index.items()):
        if word in model_word2vec:
            word2vec_embedding_metrix[i] = model_word2vec.wv[word]
        else:
            error_count += 1
    print("embedding metrix构建完成，共有未知词"+str(error_count)+"个")

    # train/test数据进行padding处理
    train_padding = tokenizer.texts_to_sequences(train_sentence)
    test_padding = tokenizer.texts_to_sequences(test_sentence)
    # maxlen根据具体需求进行更改
    train_padding = pad_sequences(train_padding, maxlen=30)
    test_padding = pad_sequences(test_padding, maxlen=30)

    return train_padding, test_padding, word2vec_embedding_metrix, vocab_size
