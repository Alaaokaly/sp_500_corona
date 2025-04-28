import os
import re  
import collections
import torch 
from torch import nn 

from torch.utils.data import TensorDataset, DataLoader

class Vocab:
    def __init__(self,tokens = [],min_freq=0,reserved_tokens = []):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line ]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(),key=lambda x:x[1], reverse=True)
         # sorted llist of unique letters/items in the corpus
         #helps to get the letter from its index/order in the array
        self.idx_to_token = list(sorted(set(['<UNK>']+ reserved_tokens + [token for token, freq in self.token_freqs if freq
        > min_freq])))
        # a dictionary that of token: index by enumarting (giving each token an index starting from 0) each letter 
        # helps y taking the letter and gives it's index 
        self.token_to_index = {token :idx for idx, token in enumerate(self.idx_to_token)}
       
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        """iterate the dictionary by letter to get the index"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_index.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        """iterrate the array by index to get the letter """
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]
    @property
    def unk(self):  
        """a fancy way to write :
               token_to_index.get(tokens, self.token_to_index['<UNK>'])
                in __getitem__() """
        return self.token_to_index['<UNK>']
    

    

class MTDeuEng(nn.Module):
    def __init__(self, batch_size, num_steps =25, num_train = 20000, num_val = 7000):
        super().__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train  = num_train
        self.num_val = num_val
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(self.get_data())
        self.train_loader = self.get_dataloader(train=True)
        self.val_loader = self.get_dataloader(train=False) 
    
    def _build_arrays(self,raw_text, src_vocab = None, tgt_vocab = None):
        def _build_array(sentences, vocab, is_tgt = False):
            pad_or_trim = lambda seq, t :(
                seq[:t] if len(seq) > t else seq + ['<pad>']*(t-len(seq)))
            sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
            if is_tgt:
                sentences= [['<bos>'] + s for s in sentences]
            if vocab is None:
                vocab =  Vocab(sentences, min_freq=5)
                array = torch.tensor([vocab[s] for s in sentences])
                valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
                return array, vocab,valid_len
        src, tgt = self._tokenize(self._preprocesing(raw_text),
                                  self.num_train + self.num_val)
        src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
        tgt_array, tgt_vocab, _ = _build_array(tgt,tgt_vocab)
        return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
        src_vocab, tgt_vocab) #index-word maps 


    def get_data(self):
        
        root = "deu.txt"
        with open (root, 'r', encoding='utf-8') as f:
             lines = f.read()
             lines = re.sub(r'CC-BY\s*(.*)'," ", lines).strip()
        modified_file = 'parallel.txt'
        with open(modified_file, 'w') as f :
            f.write(lines)
        return lines
    def _preprocesing(self, text):
        text =text.replace('\u202f', ' ').replace('\xa0', ' ')
        no_space = lambda char, prev_char : char in ',.!?' and prev_char != " "
        out = [" " + char if i > 0 and no_space(char, text[i-1]) 
               else char for i, char in enumerate(text.lower())]
        return ''.join(out)
    def _tokenize(self, text, max_examples  = None):
        src, tgt = [],[]
        for i, line in enumerate(text.split('\n')):
            if max_examples and  i > max_examples :break
            delimiters = r"[.,?!]"
            parts = line.split('\t')

            if len(parts) > 2 :
                src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
                tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
        return src, tgt 
    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)

        src_array = self.arrays[0][idx]
        tgt_array = self.arrays[1][idx]
        src_valid_len = self.arrays[2][idx]
        label_array = self.arrays[3][idx]

        dataset = TensorDataset(src_array, tgt_array, src_valid_len, label_array)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)
    
    def train_dataloader(self):
        return self.train_loader

data = MTDeuEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.get_dataloader(train=True)))
print('source:', src.type(torch.int32))
print('decoder input:', tgt.type(torch.int32))
print('source len excluding pad:', src_valid_len.type(torch.int32))
print('label:', label.type(torch.int32))