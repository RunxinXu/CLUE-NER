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
from torchcrf import CRF
import train

TEST_LOG_FN = './test_log.txt'


def test(test_dataloader, args, writer, log):
    model = train.MyNERModel(args).cuda()
    model_param = torch.load(os.path.join(args.output_dir, 'global_best_model.bin'))
    model.load_state_dict(model_param)
    model.eval()
    f1 = train.validate(model, test_dataloader, args, log, is_test=True)
    print('-----------Test set, F1-Score: %.4f-----------' % f1)
    log.write('-----------Test set, F1-Score: %.4f----------- \n' % f1)


if __name__ == '__main__':
    args = train.get_argparse().parse_args()
    tokenizer = train.MyTokenizer.from_pretrained(args.model_name)
    test_dataset = train.NERDataset(os.path.join(args.data_dir, 'test.json'), tokenizer, train=False)
    test_dataloader = train.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train.collate_fn)
    test_log = open(TEST_LOG_FN, 'w')
    writer = train.SummaryWriter(args.output_dir)
    test(test_dataloader, args, writer, test_log)
    test_log.close()
