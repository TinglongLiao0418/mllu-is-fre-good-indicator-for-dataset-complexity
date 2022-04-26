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

class RACEDatasetForRoberta(RACEDataset):
    def __init__(self, path, tokenizer, split_type="train"):
        super().__init__(path, tokenizer, split_type)

    def collate_fn(self, batch):
        input_ids = []
        attention_mask = []
        labels = []

        for example in batch:
            prompt = example['article'] + example['question']
            encoding = self.tokenizer([prompt for i in range(len(example['options']))], example['options'],
                                      return_tensors="pt", max_length=self.tokenizer.model_max_length,
                                      padding='max_length', truncation='only_first')
            input_ids.append(encoding.input_ids)
            attention_mask.append(encoding.attention_mask)
            labels.append(example['label'])

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.LongTensor(labels)
        }

class RACEDatasetForT5(RACEDataset):
    def __init__(self, path, tokenizer, split_type="train"):
        super().__init__(path, tokenizer, split_type)

    def collate_fn(self, batch):
        input_ids = []
        attention_mask = []
        labels = []

        letters = ['A', 'B', 'C', 'D', 'E']
        for example in batch:
            prompt = "Article: " + example['article'] + "Question: " +example['question'] + \
                     ' '.join([letters[i] + '. ' + example['options'][i] for i in range(len(example['options']))])
            encoding = self.tokenizer(prompt,
                                      return_tensors="pt", max_length=self.tokenizer.model_max_length - 4,
                                      padding='max_length', truncation=True)
            input_ids.append(encoding.input_ids)
            attention_mask.append(encoding.attention_mask)
            label = self.tokenizer("Answer: " + example['answer'],
                                   return_tensors="pt", max_length=4,
                                   padding='max_length', truncation=True).input_ids
            labels.append(label)

        return {
            'input_ids': torch.cat(input_ids, 0),
            'attention_mask': torch.cat(attention_mask, 0),
            'labels': torch.cat(labels, 0)
        }

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    path = '../data/RACE'
    model_name_or_path = 't5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    split_type = 'train'
    train_dataset = RACEDatasetForT5(path=path, tokenizer=tokenizer, split_type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=train_dataset.collate_fn)
    for i in tqdm(train_dataloader):
        print(i)
        break
    test_dataset = RACEDataset(path=path, tokenizer=tokenizer, split_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=train_dataset.collate_fn)
    for i in tqdm(test_dataloader):
        pass


