import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('mllu-is-fre-good-indicator-for-dataset-complexity')+1])
sys.path.insert(1, project_path)

from transformers import BertTokenizer, BertForMultipleChoice

from src.dataset import RACEDataset
from src.trainer import run_experiment

if __name__ == '__main__':
    model_name_or_path = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    train_dataset = RACEDataset(path="../../data/RACE",
                                tokenizer=tokenizer, split_type='train')
    eval_dataset = RACEDataset(path="../../data/RACE",
                               tokenizer=tokenizer, split_type='dev')
    model = BertForMultipleChoice.from_pretrained(model_name_or_path)
    run_experiment(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        data_collator=train_dataset.collate_fn,
        output_dir="log",
    )
