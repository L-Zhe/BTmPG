import  torch
from    torch import LongTensor
from    torch.utils.data import DataLoader, TensorDataset
from    .Constants import *


def create_vocab(file_list, vocab_num=-1):
    def create_corpus(file):
        with open(file, 'r') as f:
            corpus = [word.lower() for line in f.readlines() for word in line.strip('\n').split()]
        return corpus
    corpus = []
    for file in file_list:
        corpus.extend(create_corpus(file))
    
    word2index = {}; index2word = {}
    word2index[PAD_WORD] = PAD; index2word[PAD] = PAD_WORD
    word2index[UNK_WORD] = UNK; index2word[UNK] = UNK_WORD
    word2index[BOS_WORD] = BOS; index2word[BOS] = BOS_WORD
    word2index[EOS_WORD] = EOS; index2word[EOS] = EOS_WORD
    if vocab_num != -1:
        word_count = {}
        for word in corpus:
            if word_count.get(word) is None:
                word_count[word] = 1
            else:
                word_count[word] += 1
        w_count = [[word, word_count[word]] for word in word_count.keys()]
        w_count.sort(key=lambda elem: elem[1], reverse=True)
        w_count = [w_count[i][0] for i in range(min(len(w_count), vocab_num))]
    else:
        w_count = set(corpus)
    for word in w_count:
        word2index[word] = len(word2index)
        index2word[len(index2word)] = word
        
    return word2index, index2word


def lang(filelist, word2index, PAD, BOS=None, EOS=None, max_len=None):
    data = []
    for file in filelist:
        with open(file, 'r') as f:
            data.extend([[word.lower() for word in line.strip('\n').split()] for line in f.readlines()])
    if max_len is not None:
        for i in range(len(data)):
            if len(data[i]) > max_len:
                ed = data[i][-1]
                data[i] = data[i][:max_len-1]
                data[i].append(ed)
                
    def prepare_sequence(seq, word2index, max_length, PAD, BOS, EOS):
        pad_length = max_length - len(seq)
        if BOS is not None:
            seq = [BOS] + seq
        if EOS is not None:
            seq += [EOS]
        seq += [PAD] * pad_length
        
        return list(map(lambda word: word2index[UNK_WORD] 
                        if word2index.get(word) is None else word2index[word], seq))
    max_length = max([len(seq) for seq in data])
    return list(map(lambda seq: prepare_sequence(seq, word2index, max_length,
                                                 PAD, BOS, EOS), data))


def get_dataloader(source, target=None, src_inputs=None, src_outputs=None, 
                   tgt_inputs=None, tgt_outputs=None, 
                   batch_size=64, shuffle=False):
    batch_size = batch_size
    source = LongTensor(source)
    try:
        target = LongTensor(target)
        src_inputs = LongTensor(src_inputs)
        src_outputs = LongTensor(src_outputs)
        tgt_inputs = LongTensor(tgt_inputs)
        tgt_outputs = LongTensor(tgt_outputs)
        data = TensorDataset(source, target, 
                             src_inputs, src_outputs,
                             tgt_inputs, tgt_outputs)
    except:
        data = source
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def get_pretrain_dataloader(source, tgt_input, tgt_output, batch_size=64, shuffle=None):
    source = LongTensor(source)
    tgt_input = LongTensor(tgt_input)
    tgt_output = LongTensor(tgt_output)
    data = TensorDataset(source, tgt_input, tgt_output)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def translate2word(sequence, index2word):
    return [[index2word[index] for index in seq] for seq in sequence]