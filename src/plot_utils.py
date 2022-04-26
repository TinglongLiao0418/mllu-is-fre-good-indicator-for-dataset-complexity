from textstat import textstat
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_fre_acc_graph_for_model(model, dataset, save_path='./img/result.jpg', min_fre=40, max_fre=100, step_size=5):
    data = []
    for example in tqdm(dataset):
        fre_score = textstat.flesch_reading_ease(example['article'])
        logits = model(**dataset.collate_fn([example])).logits.squeeze()
        pred = logits.argmax(-1)
        result = True if pred.item() == example['label'] else False
        data.append((fre_score, result))
        break

    data.sort(key=lambda x: x[0])

    x, y = [], []

    bound = (min_fre, min_fre+step_size)
    cur_tot, cur_correct = 0, 0
    i = 0
    while i < len(data):
        f, r = data[i]
        if f > bound[1]:
            bound = (bound[0]+step_size, bound[1]+step_size)
            if cur_tot:
                x.append(sum(bound)/2)
                y.append(cur_correct/cur_tot)
            cur_tot, cur_correct = 0, 0
        else:
            cur_tot += 1
            cur_correct += int(r)
        i += 1

    print(x, y)

    plt.plot(x, y)
    plt.xticks(np.xticks(np.arange(min_fre, max_fre, step_size)))
    plt.savefig(save_path)





