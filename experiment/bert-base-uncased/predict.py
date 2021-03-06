import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('mllu-is-fre-good-indicator-for-dataset-complexity')+1])
sys.path.insert(1, project_path)

from tqdm import tqdm
import pandas as pd
from textstat import textstat
from transformers import BertTokenizer, BertForMultipleChoice
from src.dataset import RACEDataset

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = RACEDataset(path="../../data/RACE",
                               tokenizer=tokenizer, split_type='test')

    f = open('prediction_fre.csv', 'w')

    # df = pd.DataFrame(columns=['fre_score', 'is_right_prediction'])
    model = BertForMultipleChoice.from_pretrained('log/checkpoint-65898')
    for example in tqdm(test_dataset):
        fre_score = textstat.flesch_reading_ease(example['article'])
        logits = model(**test_dataset.collate_fn([example])).logits.squeeze()
        pred = logits.argmax(-1)
        result = True if pred.item() == example['label'] else False
        f.write(f"{fre_score}, {result}\n")

    f.close()

    # df.to_csv('prediction_fre.csv')