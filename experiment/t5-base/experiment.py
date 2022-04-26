import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('mllu-is-fre-good-indicator-for-dataset-complexity')+1])
sys.path.insert(1, project_path)

from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.dataset import RACEDatasetForT5
from src.trainer import run_experiment

if __name__ == '__main__':
    model_name_or_path = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    train_dataset = RACEDatasetForT5(path="../../data/RACE",
                                     tokenizer=tokenizer, split_type='train')
    eval_dataset = RACEDatasetForT5(path="../../data/RACE",
                                    tokenizer=tokenizer, split_type='dev')
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    run_experiment(
        model=model,
        learning_rate=1e-4,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn,
        output_dir="log",
    )
