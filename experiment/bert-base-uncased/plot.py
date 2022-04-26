import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('mllu-is-fre-good-indicator-for-dataset-complexity')+1])
sys.path.insert(1, project_path)

from src.plot_utils import plot_fre_acc_graph_for_model
from transformers import BertTokenizer, BertForMultipleChoice
from src.dataset import RACEDataset

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = RACEDataset(path="../../data/RACE",
                               tokenizer=tokenizer, split_type='test')

    model = BertForMultipleChoice.from_pretrained('log/checkpoint-65000')
    plot_fre_acc_graph_for_model(model, test_dataset, save_path='fig/result.jpg')