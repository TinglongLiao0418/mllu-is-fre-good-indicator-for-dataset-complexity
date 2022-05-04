import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('mllu-is-fre-good-indicator-for-dataset-complexity')+1])
sys.path.insert(1, project_path)

from transformers import RobertaTokenizer, RobertaForMultipleChoice

from src.dataset import RACEDatasetForRoberta
from src.trainer import run_experiment

if __name__ == '__main__':
    model_name_or_path = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    train_dataset = RACEDatasetForRoberta(path="../../data/RACE",
                                tokenizer=tokenizer, split_type='train')
    eval_dataset = RACEDatasetForRoberta(path="../../data/RACE",
                               tokenizer=tokenizer, split_type='dev')
    model = RobertaForMultipleChoice.from_pretrained(model_name_or_path)
    run_experiment(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epoch=8,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        data_collator=train_dataset.collate_fn,
        output_dir="log",
    )
