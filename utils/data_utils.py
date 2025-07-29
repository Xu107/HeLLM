import pandas as pd
import numpy as np
import torch
import json
import time
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import os
all_items_num=0

import config
args=config.parse_args_llama()

class SequentialDataset(Dataset):
    def __init__(self, dataset, maxlen):
        super(SequentialDataset, self).__init__()
        self.dataset_path = f"./datasets/sequential/{dataset}"
        self.maxlen = maxlen

        self.trainData, self.valData, self.testData = [], {}, {}
        self.n_user, self.m_item = 0, 0

        with open(os.path.join(self.dataset_path,f'{dataset}.txt'), 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                # user, items = int(line[0]), [int(item)+1 for item in line[1:]]#Very important! 在序列推荐，user为了计数遍历不加1，从0开始，item由于SASRec等下标从1开始
                user, items = int(line[0]), [int(item) for item in line[1:]]#Very important! 在序列推荐，user为了计数遍历不加1，从0开始，item由于SASRec等下标从1开始
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

                if len(items)>maxlen:
                    items=items[-maxlen:]

                if len(items) >= 3:
                    train_items = items[:-2]
                    length = len(train_items)
                    for t in range(length):
                        #0:[[],1] if t==0:continue?
                        if t==0:
                            continue
                        self.trainData.append([train_items[:-length + t], train_items[-length + t]])
                    self.valData[user] = [items[:-2], items[-2]]
                    self.testData[user] = [items[:-1], items[-1]]
                else:
                    for t in range(len(items)):
                        self.trainData.append([items[:-len(items) + t], items[-len(items) + t]])
                    self.valData[user] = []
                    self.testData[user] = []

        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1

    def __getitem__(self, idx):
        seq, label = self.trainData[idx]
        return seq, label

    def __len__(self):
        return len(self.trainData)

@dataclass
class SequentialCollator:
    def __call__(self, batch) -> dict:
        seqs, labels = zip(*batch)
        max_len = max(max([len(seq) for seq in seqs]), 2)
        inputs = [[0] * (max_len - len(seq)) + seq for seq in seqs]
        inputs_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]
        labels = [[label] for label in labels]
        inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels
        }