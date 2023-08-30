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
# MAX_LEN = 256

def create_train_data_translate(file_path, MAX_LEN = 512):
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
        en_sent, cn_sent, _ = line.split('\t')  # Hi.嗨。
        # 构造一个新的翻译格式...
        # {"messages": [{"role": "system", "content": "你是一个汉语语法纠错器。"},
        #               {"role": "user", "content": "检测这个句子的语法错误：这件事对我们大家当时震动很大。"},
        #               {"role": "assistant", "content": "这件事当时对我们大家震动很大。"}]}
        str = '{"messages": [{"role": "system", "content": "你是一个中英文翻译器。"}, {"role": "user", "content": "%s" }, {"role": "assistant", "content": "%s"}]}\n' % (cn_sent, en_sent)
        pairs.append(str)
    # 生成train_data.jsonl
    with open("train_data.jsonl", 'wt', encoding='utf-8') as f:
        f.writelines(pairs)
        f.close()


def create_dataset(file_path, MAX_LEN = 256):
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
    #pairs = (['wait', '!'], ['等', '等', '！']), (['wait', '!'], ['等', '一', '下', '！']), (['begin', '.'], ['开', '始', '！']), (['hello', '!'], ['你', '好', '。']), (['i', 'try', '.'], ['我', '试', '试', '。']), (['i', 'won', '!'], ['我', '赢', '了', '。']), (['oh', 'no', '!'], ['不', '会', '吧', '。']), (['cheers', '!'], ['乾', '杯', '!']), (['got', 'it', '?'], ['知', '道', '了', '没', '有', '？']), (['got', 'it', '?'], ['懂', '了', '吗', '？']), (['got', 'it', '?'], ['你', '懂', '了', '吗', '？']), (['he', 'ran', '.'], ['他', '跑', '了', '。']), (['hop', 'in', '.'], ['跳', '进', '来', '。']), (['i', 'know', '.'], ['我', '知', '道', '。']), (['i', 'quit', '.'], ['我', '退', '出', '。']), (['i', 'quit', '.'], ['我', '不', '干', '了', '。']), (['i', 'm', 'ok', '.'], ['我', '沒', '事', '。']), (['i', 'm', 'up', '.'], ['我', '已', '经', '起', '来', '了', '。']), (['listen', '.'], ['听', '着', '。']), (['no', 'way', '!'], ['不', '可', '能', '！']), (['no', 'way', '!'], ['没', '门', '！']), (['really', '?'], ['你', '确', '定', '？']), (['thanks', '!'], ['谢', '谢', '！']), (['try', 'it', '.'], ['试', '试', '吧', '。']), (['we', 'try', '.'], ['我', '们', '来', '试', '试', '。']), (['why', 'me', '?'], ['为', '什', '么', '是', '我', '？']), (['ask', 'tom', '.'], ['去', '问', '汤', '姆', '。']), (['awesome', '!'], ['好', '棒', '！']), (['be', 'calm', '.'], ['冷', '静', '点', '。']), (['be', 'fair', '.'], ['公', '平', '点', '。']), (['be', 'kind', '.'], ['友', '善', '点', '。']), (['be', 'kind', '.'], ['友', '好', '點', '。']), (['be', 'nice', '.'], ['和', '气', '点', '。']), (['be', 'nice', '.'], ['友', '善', '点', '。']), (['call', 'me', '.'], ['联', '系', '我', '。']), (['call', 'us', '.'], ['联', '系', '我', '们', '。']), (['come', 'in', '.'], ['进', '来', '。']), (['get', 'tom', '.'], ['找', '到', '汤', '姆', '。']), (['get', 'out', '!'], ['滾', '出', '去', '！']), (['get', 'out', '!'], ['出', '去', '！']), (['go', 'away', '!'], ['走', '開', '！']), (['go', 'away', '!'], ['滾', '！']), (['go', 'away', '.'], ['走', '開', '！']), (['go', 'home', '.'], ['回', '家', '。']), (['go', 'home', '.'], ['回', '家', '吧', '。']), (['goodbye', '!'], ['再', '见', '！']), (['goodbye', '!'], ['告', '辞', '！']), (['hang', 'on', '!'], ['坚', '持', '。']), (['hang', 'on', '!'], ['等', '一', '下', '！']), (['hang', 'on', '.'], ['坚', '持', '。']), (['he', 'came', '.'], ['他', '来', '了', '。']), (['he', 'runs', '.'], ['他', '跑', '。']), (['help', 'me', '.'], ['帮', '我', '一', '下', '。']), (['help', 'us', '.'], ['帮', '帮', '我', '们', '吧', '！']), (['hit', 'tom', '.'], ['去', '打', '汤', '姆', '。']), (['hold', 'on', '.'], ['坚', '持', '。']), (['hug', 'tom', '.'], ['抱', '抱', '汤', '姆', '！']), (['hug', 'tom', '.'], ['请', '抱', '紧', '汤', '姆', '。']), (['i', 'agree', '.'], ['我', '同', '意', '。']), (['i', 'm', 'hot', '.'], ['我', '觉', '得', '很', '热', '。']), (['i', 'm', 'ill', '.'], ['我', '生', '病', '了', '。']), (['i', 'm', 'sad', '.'], ['我', '很', '难', '过', '。']), (['i', 'm', 'shy', '.'], ['我', '很', '害', '羞', '。']), (['i', 'm', 'wet', '.'], ['我', '濕', '了', '。']), (['it', 's', 'ok', '.'], ['没', '关', '系', '。']), (['it', 's', 'me', '.'], ['是', '我', '。']), (['join', 'us', '.'], ['来', '加', '入', '我', '们', '吧', '。']), (['keep', 'it', '.'], ['留', '着', '吧', '。']), (['kiss', 'me', '.'], ['吻', '我', '。']), (['perfect', '!'], ['完', '美', '！']), (['see', 'you', '.'], ['再', '见', '！']), (['shut', 'up', '!'], ['閉', '嘴', '！']), (['skip', 'it', '.'], ['不', '管', '它', '。']), (['take', 'it', '.'], ['拿', '走', '吧', '。']), (['tell', 'me', '.'], ['告', '诉', '我', '！']), (['tom', 'won', '.'], ['汤', '姆', '胜', '利', '了', '。']), (['wake', 'up', '!'], ['醒', '醒', '！']), (['wash', 'up', '.'], ['去', '清', '洗', '一', '下', '。']), (['we', 'know', '.'], ['我', '们', '知', '道', '。']), (['welcome', '.'], ['欢', '迎', '。']), (['who', 'won', '?'], ['谁', '赢', '了', '？']), (['why', 'not', '?'], ['为', '什', '么', '不', '？']), (['you', 'run', '.'], ['你', '跑', '。']), (['you', 'win', '.'], ['算', '你', '狠', '。']), (['back', 'off', '!'], ['往', '后', '退', '点', '。']), (['back', 'off', '!'], ['后', '退', '！']), (['back', 'off', '.'], ['往', '后', '退', '点', '。']), (['be', 'still', '.'], ['静', '静', '的', '，', '别', '动', '。']), (['beats', 'me', '.'], ['我', '一', '无', '所', '知', '。']), (['cuff', 'him', '.'], ['把', '他', '铐', '上', '。']), (['drive', 'on', '.'], ['往', '前', '开', '。']), (['get', 'away', '!'], ['走', '開', '！']), (['get', 'away', '!'], ['滾', '！']), (['get', 'down', '!'], ['趴', '下', '！']), (['get', 'lost', '!'], ['滾', '！']), (['get', 'lost', '!'], ['滚', '。'...

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



if __name__ == '__main__':
    create_train_data_translate("./cmn-eng/cmn.txt")