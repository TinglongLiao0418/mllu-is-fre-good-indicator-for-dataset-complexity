import os
import json

import torch
from torch.utils.data import Dataset

class RACEDataset(Dataset):
    def __init__(self, path, tokenizer, split_type="train"):
        self.path = path
        self.tokenizer = tokenizer
        self.split_type = split_type

        self._prepaer_data()

    def _prepaer_data(self):
        self.data = []

        for level in ["middle", "high"]:
            path = os.path.join(self.path, self.split_type, level)
            for d in os.listdir(path):
                file = open(os.path.join(path, d))
                test = json.load(file)
                for i in range(len(test['questions'])):
                    example = {
                        'article': test['article'],
                        'question': test['questions'][i],
                        'options': test['options'][i],
                        'answer': test['answers'][i],
                    }
                    answer_to_index = {'A':0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                    example['label'] = answer_to_index[example['answer']]
                    self.data.append(example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []

        for example in batch:
            prompt = example['article'] + example['question']
            encoding = self.tokenizer([prompt for i in range(len(example['options']))], example['options'],
                                      return_tensors="pt", max_length=self.tokenizer.model_max_length,
                                      padding='max_length', truncation='only_first')
            input_ids.append(encoding.input_ids)
            attention_mask.append(encoding.attention_mask)
            token_type_ids.append(encoding.token_type_ids)
            labels.append(example['label'])

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'token_type_ids': torch.stack(token_type_ids),
            'labels': torch.LongTensor(labels)
        }

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer
    path = '../data/RACE'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    split_type = 'train'
    train_dataset = RACEDataset(path=path, tokenizer=tokenizer, split_type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=train_dataset.collate_fn)
    for i in tqdm(train_dataloader):
        pass

    test_dataset = RACEDataset(path=path, tokenizer=tokenizer, split_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=train_dataset.collate_fn)
    for i in tqdm(test_dataloader):
        pass


