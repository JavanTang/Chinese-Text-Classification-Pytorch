'''
@Author: TangZhiFeng
@Data: Do not edit
@LastEditors: TangZhiFeng
@LastEditTime: 2020-04-16 09:59:59
@Description: 
'''
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

class Test_utils(unittest.TestCase):
    def test_sentance2ids(self):
        result = sentance2ids('传蔡卓妍与英皇约满后跳槽金牌大风', config)
        self.assertTrue(isinstance(result, list))


    def test_predict(self):
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
        _test_sentance = sentance2ids(['2岁男童爬窗台不慎7楼坠下获救(图)'], config)
        _test_sentance = build_iterator(_test_sentance, config)
        for X, y in _test_sentance:
            result = model(X)
            result = torch.max(result.data, 1)[1].cpu().numpy()
        result = list(result)
        print(result[0])
        self.assertTrue(isinstance(result, list))


if __name__ == '__main__':
    unittest.main()
