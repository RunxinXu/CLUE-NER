from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json

# 简单检查了数据
# 一共有十个类 而且不存在overlap or nested的现象

# use BIOS schema
# KIND_i => B(i), I(i+10), S(i+20)
# O => 30
# TARGET_PAD => 31
KIND = ['name', 'book', 'organization', 'company', 'game', 'address', 'scene', 'government', 'position', 'movie'] #len=10
KIND2ID = dict()
for i in range(len(KIND)):
    KIND2ID[KIND[i]] = i
TARGET_PAD = 31 # i.e., if the label is 31, we can ignore it and do not need to calculate the according loss
CONTEXT_PAD = 0

class MyTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=True, **kwargs):
        super().__init__(vocab_file=vocab_file, do_lower_case=do_lower_case, **kwargs)

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens

class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer): 
        self.data = []
        self.B = 0
        self.I = 10
        self.S = 20
        
        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                text = d['text']
                tokens = tokenizer.tokenize(text) # list
                tokens = tokenizer.convert_tokens_to_ids(tokens) # list
                labels = [30] * len(tokens) # 30 means O, i.e., not an entity
                
                label = d['label']
                for kind, entities in label.items():
                    kind_id = KIND2ID[kind]
                    for entity, spans in entities.items():
                        for span in spans:
                            begin_idx, end_idx = span
                            if begin_idx == end_idx: # single
                                labels[begin_idx] = kind_id + self.S
                            else:
                                labels[begin_idx] = kind_id + self.B
                                for i in range(begin_idx+1, end_idx+1):
                                    labels[i] = kind_id + self.I
                    
                # X => [cls] X [sep]
                tokens = tokenizer.build_inputs_with_special_tokens(tokens)
                labels = [TARGET_PAD] + labels + [TARGET_PAD]
                assert len(tokens) == len(labels)
                
                # to Tensor
                tokens = torch.LongTensor(tokens)
                labels = torch.LongTensor(labels)
                
                self.data.append({
                    'tokens': tokens,
                    'labels': labels,
                    'length': tokens.size(0)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
def collate_fn(batch):

    lengths = torch.LongTensor([sample['length'] for sample in batch])
    tokens = [torch.LongTensor(sample['tokens']) for sample in batch]
    labels = [torch.LongTensor(sample['labels']) for sample in batch]

    tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=CONTEXT_PAD)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=TARGET_PAD)

    data = {
        'tokens': tokens,
        'labels': labels,
        'lengths': lengths
    }
    return data

class MyNERModel(nn.Module):
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext-large'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name) # BertModel
        self.total_kinds = 31
        self.predict = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, self.total_kinds),
        )
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.loss = nn.NLLLoss(ignore_index=TARGET_PAD)

    def forward(self, tokens, labels=None):
        # tokens: bsz * length
        mask = tokens != CONTEXT_PAD
        output = self.bert(input_ids=tokens, attention_mask=mask)
        output = output[0] # bsz * length * dim
        logits = self.predict(output)
        prediction = self.logsoftmax(logits)
        
        if labels is None:
            return prediction

        prediction = prediction[:, 1:-1, :] # skip [cls] and [sep]
        labels = labels[:, 1:-1] 
        loss = self.loss(prediction.reshape(-1, self.total_kinds), labels.reshape(-1))
        return prediction, loss

if __name__ == '__main__':
    tokenizer = MyTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large") # MyTokenizer
    dataset = NERDataset('data/train.json', tokenizer)
    data_loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)
    model = MyNERModel().cuda()
    for batch in data_loader:
        prediction, loss = model(tokens=batch['tokens'].cuda(), labels=batch['labels'].cuda())
        print(loss)
        input()

