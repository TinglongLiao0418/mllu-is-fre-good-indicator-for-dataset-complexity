from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer,Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments


def compute_metric(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    return {
        'accuracy': accuracy_score(labels, preds),
    }

def compute_metric_t5(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.logits.argmax(-1)

    acc = (labels == preds).all(dim=-1).int().sum().item() / labels.size(0)

    return {
        'accuracy': acc
    }


def run_experiment(model, train_dataset, eval_dataset, data_collator, output_dir='log', learning_rate=1e-5,
                   gradient_accumulation_steps=4, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                   epoch=6, seed=42):
    train_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        evaluation_strategy='epoch',
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epoch,
        save_strategy='epoch',
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metric
    )

    trainer.train()

def run_generative_experiment(model, train_dataset, eval_dataset, data_collator, output_dir='log', learning_rate=1e-5,
                              gradient_accumulation_steps=4, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                              evaluation_strategy="epoch", eval_steps=1e5, eval_accumulation_steps=None, epoch=6, seed=42):
    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        num_train_epochs=epoch,
        save_strategy='epoch',
        seed=seed
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metric_t5
    )

    trainer.train()