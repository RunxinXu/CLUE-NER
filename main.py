from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import json
import argparse
import os
from collections import defaultdict
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

# 简单检查了数据
# 一共有十个类 而且不存在overlap or nested的现象

# use BIOS schema
# KIND_i => B(i), I(i+10), S(i+20)
# O => 30
# TARGET_PAD => 31
KIND = ['name', 'book', 'organization', 'company', 'game', 'address', 'scene', 'government', 'position', 'movie'] #len=10
KIND2ID = dict()
ID2KIND = dict()
for i in range(len(KIND)):
    KIND2ID[KIND[i]] = i
    ID2KIND[i] = KIND[i]
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
    def __init__(self, data_path, tokenizer, train=True): 
        self.data = []
        self.B = 0
        self.I = 10
        self.S = 20
        
        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                text = d['text']
                tokens = tokenizer.tokenize(text) # list
                raw_token = tokens
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
                    'length': tokens.size(0),
                    'raw_label': label,
                    'raw_token': raw_token
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

    if batch[0]['raw_label'] is not None:
        data['raw_label'] = [sample['raw_label'] for sample in batch]
        data['raw_token'] = [sample['raw_token'] for sample in batch]
    
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

def train(train_dataset, args, writer):
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=0)
    global_best_f1 = 0
    for fold_cnt, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        train_data = [train_dataset[i] for i in train_idx]
        val_data = [train_dataset[i] for i in val_idx]
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        model = MyNERModel(args).cuda()
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        global_step = 0
        fold_best_f1 = 0

        print('****************Fold %d*****************' % fold_cnt)
        for epoch in range(args.epochs):
            print('Epoch %d' % epoch)
            for batch in tqdm(train_dataloader):
                prediction, loss = model(tokens=batch['tokens'].cuda(), labels=batch['labels'].cuda())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Fold_'+str(fold_cnt)+'_Train/Loss', loss.item(), global_step)
                global_step += 1
            
            # eval
            model.eval()
            f1 = validate(model, val_dataloader, args)
            writer.add_scalar('Val/F1', f1, epoch)
            model.train()

            if f1 > fold_best_f1:
                fold_best_f1 = f1
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'fold_'+str(fold_cnt)+'_best_model.bin'))
                if f1 > global_best_f1:
                    global_best_f1 = f1
                    torch.save(model.state_dict(), os.path.join(args.output_dir, 'global_best_model.bin'))
                    print('-----------Global Best: Fold %d, F1-Score: %.4f-----------' % (fold_cnt, f1))


def validate(model, val_dataloader, args):
    pred = []
    gold = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            prediction = model(tokens=batch['tokens'].cuda(), labels=None)
            prediction = prediction[:, 1:-1, :]
            prediction = torch.argmax(prediction, dim=-1).cpu().numpy() # bsz * length
            bsz, length = prediction.shape
            for i in range(bsz):
                cur_result = {}
                for entity_type in KIND:
                    cur_result[entity_type] = {}
                raw_token = batch['raw_token'][i]
                gold.append(batch['raw_label'][i])

                j = 0
                while j < len(raw_token):
                    if prediction[i][j] >= 20 and prediction[i][j] < 30: # S
                        entity_name = raw_token[j]
                        entity_type = ID2KIND[prediction[i][j]-20]
                        if entity_name not in cur_result[entity_type]:
                            cur_result[entity_type][entity_name] = []
                        cur_result[entity_type][entity_name].append([j, j])
                        j += 1
                    elif prediction[i][j] < 10: # B
                        entity_type = ID2KIND[prediction[i][j]]
                        shouldbe = KIND2ID[entity_type] + 10 # should be I-same kind
                        new_j = j + 1
                        while new_j < len(raw_token) and prediction[i][new_j] == shouldbe:
                            new_j += 1
                        if new_j > j + 1:
                            entity_name = ''.join(raw_token[j:new_j])
                            if entity_name not in cur_result[entity_type]:
                                cur_result[entity_type][entity_name] = []
                            cur_result[entity_type][entity_name].append([j, new_j-1])
                            j = new_j
                        else:
                            # not valid
                            j = new_j
                    else: # I or O
                        j += 1

                pred.append(cur_result)

    _, macro_f1 = get_f1_score(pred, gold)
    return macro_f1


def test(test_dataloader, args, writer):
    model = MyNERModel(args).cuda()
    model_param = torch.load(os.path.join(args.output_dir, 'global_best_model.bin'))
    model.load_state_dict(model_param)
    model.eval()
    f1 = validate(model, test_dataloader, args)
    print('-----------Test set, F1-Score: %.4f-----------' % f1)


def get_f1_score_label(pred, gold, label="organization"):
    """
    打分函数
    """
    TP = 0
    FP = 0
    FN = 0
    for p, g in zip(pred, gold):

        p = p.get(label, {}).keys()
        g = g.get(label, {}).keys()
        for i in p:
            if i in g:
                TP += 1
            else:
                FP += 1
        for i in g:
            if i not in p:
                FN += 1

    p = TP / (TP + FP + 1e-20)
    r = TP / (TP + FN + 1e-20)
    f = 2 * p * r / (p + r + 1e-20)
    print('label: {}\nTP: {}\tFP: {}\tFN: {}'.format(label, TP, FP, FN))
    print('P: {:.2f}\tR: {:.2f}\tF1: {:.2f}'.format(p, r, f))
    print()
    return f


def get_f1_score(pred, gold):
    f_score = {}
    labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    sum = 0
    for label in labels:
        f = get_f1_score_label(pred, gold, label=label)
        f_score[label] = f
        sum += f
    avg = sum / (len(labels) + 1e-20)
    return f_score, avg

def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='hfl/chinese-roberta-wwm-ext-large', type=str,
                        help="Model name")
    parser.add_argument("--data_dir", default='data', type=str,
                        help="Data folder", )
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size" )
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="Batch size" )
    parser.add_argument("--epochs", default=50, type=int,
                        help="Output folder", )
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output folder", )
    parser.add_argument("--load_model", default=None, type=str,
                        help="Load model", )
    parser.add_argument("--n_splits", default=5, type=int,
                        help="n_splits", )
    return parser

if __name__ == '__main__':
    args = get_argparse().parse_args()
    tokenizer = MyTokenizer.from_pretrained(args.model_name)
    train_dataset = NERDataset(os.path.join(args.data_dir, 'train.json'), tokenizer)
    #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataset = NERDataset(os.path.join(args.data_dir, 'test.json'), tokenizer, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    #model = MyNERModel(args).cuda()
    #if args.load_model is not None:
    #    model.load_state_dict(torch.load(args.load_model))
    writer = SummaryWriter(args.output_dir)
    train(train_dataset, args, writer)
    test(test_dataloader, args, writer)

