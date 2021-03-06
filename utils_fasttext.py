# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """构建词袋
    
    Arguments:
        file_path {str}} -- 文本的路径
        tokenizer {func} -- 构建词的方法,两种方法:分词,分字符
        max_size {int} -- 一个词最大的数组表示长度
        min_freq {int} -- 最小的词长度
    
    Returns:
        dict -- 词袋的对应关系,例如 {"士兵":[1,0,0,0,1,...,1]}
    """
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



def sentance2ids(contents: list, config, ues_word=False, pad_size=32):
    """从句子到向量
    
    Arguments:
        content {list} -- 文本内容list
        config {配置}} -- 配置文件
    
    Keyword Arguments:
        ues_word {bool} -- 使用什么来配置 (default: {False})
        pad_size {int} -- 每个词填充的大小 (default: {32})
    
    Returns:
        list -- 向量
    """
    words_line = []
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def biGramHash(sequence, t, buckets):
        """生成两个字符在一起的hash值<===它的标题是这个意思,但是我感觉这个仅仅只是做了一个字符的hash值
        
        Arguments:
            sequence {list}} -- 字符转化成为了id之后的list
            t {int} -- 位置信息,就是对第几个位置做
            buckets {int} -- 长度
        
        Returns:
            int -- 生成的hash值
        """
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        """生成三个字符在一起的hash值<===它的标题是这个意思,但是我感觉这个仅仅只是做了两个字符的hash值
        
        Arguments:
            sequence {list}} -- 字符转化成为了id之后的list
            t {int} -- 位置信息,就是对第几个位置做
            buckets {int} -- 长度
        
        Returns:
            int -- 生成的hash值
        """
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets
    
    result = []
    for line in contents:
        lin = line.strip()
        assert(len(lin)!=0)
        words_line = []
        token = tokenizer(line)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                # extend等同于+, 同时这里是给token填充PAD字符
                token.extend([PAD] * (pad_size - len(token)))
            else:
                # 如果大于pad_size就直接截断
                token = token[:pad_size]
                seq_len = pad_size
            # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))

        # fasttext ngram
        buckets = config.n_gram_vocab
        bigram = []
        trigram = []
        # ------ngram------
        for i in range(pad_size):
            # 这个有什么意义?   貌似只有biGramHash只是求得了一个的hash值,triGramHash求得了两个的哈希值
            bigram.append(biGramHash(words_line, i, buckets))
            trigram.append(triGramHash(words_line, i, buckets))
        # -----------------
        result.append((words_line, -1, seq_len, bigram, trigram))
    return result  # [([...], 0), ([...], 1), ...]

    

def build_dataset(config, ues_word):
    # 查看是使用分词的词,还是使用字符
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    # 判断是不是有了词表
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        # 如果没有词表就把用训练数据重新训练
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def biGramHash(sequence, t, buckets):
        """生成两个字符在一起的hash值<===它的标题是这个意思,但是我感觉这个仅仅只是做了一个字符的hash值
        
        Arguments:
            sequence {list}} -- 字符转化成为了id之后的list
            t {int} -- 位置信息,就是对第几个位置做
            buckets {int} -- 长度
        
        Returns:
            int -- 生成的hash值
        """
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        """生成三个字符在一起的hash值<===它的标题是这个意思,但是我感觉这个仅仅只是做了两个字符的hash值
        
        Arguments:
            sequence {list}} -- 字符转化成为了id之后的list
            t {int} -- 位置信息,就是对第几个位置做
            buckets {int} -- 长度
        
        Returns:
            int -- 生成的hash值
        """
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):    # tqdm将f转化成为list
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        # extend等同于+, 同时这里是给token填充PAD字符
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        # 如果大于pad_size就直接截断
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # fasttext ngram
                buckets = config.n_gram_vocab
                bigram = []
                trigram = []
                # ------ngram------
                for i in range(pad_size):
                    # 这个有什么意义?   貌似只有biGramHash只是求得了一个的hash值,triGramHash求得了两个的哈希值
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))
                # -----------------
                contents.append((words_line, int(label), seq_len, bigram, trigram))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches =1 if len(batches) // batch_size == 0 else len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数 
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # xx = [xxx[2] for xxx in datas]
        # indexx = np.argsort(xx)[::-1]
        # datas = np.array(datas)[indexx]
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == "__main__":
    '''提取预训练词向量'''
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/vocab.embedding.sougou"
    word_to_id = pkl.load(open(vocab_dir, 'rb'))
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
