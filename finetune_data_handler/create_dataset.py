# -*- coding: utf-8 -*-
# @Time    : 2021/8/1 19:16
# @Author  : He Ruizhi
# @File    : machine_translation.py
# @Software: PyCharm

import paddle
import paddle.nn.functional as F
import re
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
print(paddle.__version__)  # 2.1.0

# 设置训练句子最大长度，用于筛选数据集中的部分数据
MAX_LEN = 12


def create_dataset(file_path):
    """
    构建机器翻译训练数据集

    :param file_path: 训练数据路径
    :return:
    train_en_sents：由数字ID组成的英文句子
    train_cn_sents：由数字ID组成的中文句子
    train_cn_label_sents：由数字ID组成的中文词汇标签
    en_vocab：英文词表
    cn_vocab：中文词表
    """
    with open(file_path, 'rt', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    # 设置正则匹配模板，用于从英文句子中提取单词
    words_re = re.compile(r'\w+')

    # 将训练数据文件中的中文和英文句子全部提取出来
    pairs = []
    for line in lines:
        en_sent, cn_sent, _ = line.split('\t')
        pairs.append((words_re.findall(en_sent.lower())+[en_sent[-1]], list(cn_sent)))

    # 从原始训练数据中筛选出一部分数据用来训练模型
    # 实际训练神经网络翻译机时数据量肯定是越多越好，不过本文只选取长度小于10的句子
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) < MAX_LEN and len(pair[1]) < MAX_LEN:
            filtered_pairs.append(pair)

    # 创建中英文词表，将中文和因为句子转换成词的ID构成的序列
    # 此外须在词表中添加三个特殊词：<pad>用来对短句子进行填充；<bos>表示解码时的起始符号；<eos>表示解码时的终止符号
    # 在实际任务中，一般还会需要指定<unk>符号表示在词表中未出现过的词，并在构造训练集时有意识地添加<unk>符号，使模型能够处理相应情况
    en_vocab = {}
    cn_vocab = {}
    en_vocab['<pad>'], en_vocab['<bos>'], en_vocab['<eos>'] = 0, 1, 2
    cn_vocab['<pad>'], cn_vocab['<bos>'], cn_vocab['<eos>'] = 0, 1, 2
    en_idx, cn_idx = 3, 3
    for en, cn in filtered_pairs:
        for w in en:
            if w not in en_vocab:
                en_vocab[w] = en_idx
                en_idx += 1
        for w in cn:
            if w not in cn_vocab:
                cn_vocab[w] = cn_idx
                cn_idx += 1

    # 使用<pad>符号将短句子填充成长度一致的句子，便于使用批量数据训练模型
    # 同时根据词表，创建一份实际的用于训练的用numpy array组织起来的数据集
    padded_en_sents = []
    padded_cn_sents = []
    # 训练过程中的预测的目标，即每个中文的当前词去预测下一个词是什么词
    padded_cn_label_sents = []
    for en, cn in filtered_pairs:
        padded_en_sent = en + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(en))
        padded_cn_sent = ['<bos>'] + cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn))
        padded_cn_label_sent = cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn) + 1)

        # 根据词表，将相应的单词转换成数字ID
        padded_en_sents.append([en_vocab[w] for w in padded_en_sent])
        padded_cn_sents.append([cn_vocab[w] for w in padded_cn_sent])
        padded_cn_label_sents.append([cn_vocab[w] for w in padded_cn_label_sent])

    # 将训练数据用numpy array组织起来
    train_en_sents = np.array(padded_en_sents, dtype='int64')
    train_cn_sents = np.array(padded_cn_sents, dtype='int64')
    train_cn_label_sents = np.array(padded_cn_label_sents, dtype='int64')

    return train_en_sents, train_cn_sents, train_cn_label_sents, en_vocab, cn_vocab
