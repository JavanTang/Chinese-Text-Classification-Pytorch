import numpy as np
import os
import pickle as pkl
import torch
import unittest   # The test framework
from models import FastText
from utils_fasttext import sentance2ids, build_iterator
config = FastText.YnynConfig()
MAX_VOCAB_SIZE = 10000  # 词表长度限制


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic
ues_word = False
if ues_word:
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
else:
    tokenizer = lambda x: [y for y in x]  # char-level
if os.path.exists(config.vocab_path):
    vocab = pkl.load(open(config.vocab_path, 'rb'))
else:
    vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(vocab, open(config.vocab_path, 'wb'))
config.n_vocab = len(vocab)
model = FastText.Model(config)
map_location=torch.device('cpu')
model.load_state_dict(torch.load(config.save_path, map_location=map_location))
_test_sentance = sentance2ids(['北上资金卷土重来：热捧业绩预增 大幅增仓农业股'], config)
_test_sentance = build_iterator(_test_sentance, config)
for X, y in _test_sentance:
    result = model(X)
    result = torch.max(result.data, 1)[1].cpu().numpy()
result = list(result)

classify = [
    'finance',
    'realty',
    'stocks',
    'education',
    'science',
    'society',
    'politics',
    'sports',
    'game',
    'entertainment'
]

print(classify[result[0]])