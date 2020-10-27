from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import json
import argparse
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

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
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.model_name) # BertModel
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

def train(model, train_dataloader, eval_dataloader, args, writer):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    global_step = 0
    best_f1 = 0
    for epoch in range(args.epochs):
        for batch in tqdm(train_dataloader):
            prediction, loss = model(tokens=batch['tokens'].cuda(), labels=batch['labels'].cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), global_step)
            global_step += 1
        
        # eval
        # f1 = validate(model, eval_dataloader, args)
        f1 = 1.0
        writer.add_scalar('Eval/F1', f1, epoch)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.bin'))


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='hfl/chinese-roberta-wwm-ext-large', type=str,
                        help="Model name")
    parser.add_argument("--data_dir", default='data', type=str,
                        help="Data folder", )
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size" )
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="Batch size" )
    parser.add_argument("--epochs", default=50, type=int,
                        help="Output folder", )
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output folder", )
    parser.add_argument("--load_model", default=None, type=str,
                        help="Load model", )
    return parser

if __name__ == '__main__':
    args = get_argparse().parse_args()
    tokenizer = MyTokenizer.from_pretrained(args.model_name)
    train_dataset = NERDataset(os.path.join(args.data_dir, 'train.json'), tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataset = NERDataset(os.path.join(args.data_dir, 'test.json'), tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = MyNERModel(args).cuda()
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
    writer = SummaryWriter(args.output_dir)
    train(model, train_dataloader, eval_dataloader, args, writer)

