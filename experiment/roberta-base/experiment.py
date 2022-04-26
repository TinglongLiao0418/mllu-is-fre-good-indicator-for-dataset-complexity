import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('mllu-is-fre-good-indicator-for-dataset-complexity')+1])
sys.path.insert(1, project_path)

from transformers import RobertaTokenizer, RobertaForMultipleChoice

from src.dataset import RACEDataset
from src.trainer import run_experiment

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_dataset = RACEDataset(path="../../data/RACE",
                                tokenizer=tokenizer, split_type='train')
    eval_dataset = RACEDataset(path="../../data/RACE",
                               tokenizer=tokenizer, split_type='dev')
    model = RobertaForMultipleChoice.from_pretrained('roberta-base')
    run_experiment(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn,
        output_dir="log",
    )
