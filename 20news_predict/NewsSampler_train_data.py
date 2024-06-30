import random
from sklearn.datasets import fetch_20newsgroups
import numpy as np


def prepare_query(size=5000, seed=0):
    newsgroups_train = fetch_20newsgroups(subset='train')
    assert size < len(newsgroups_train.data) and size != 0 and size is not None

    random.seed(0)
    random_indice = random.sample(range(len(newsgroups_train.data)), size)

    train_X = [newsgroups_train.data[i] for i in random_indice]
    train_y = newsgroups_train.target[random_indice]
    print("sample labels (first 10):\n%s" % str(train_y[:10]))
    return train_X, train_y


def save_ground_truth(q_size):
    _, y = prepare_query(q_size)

    np.savetxt("20News_%d.csv" % q_size, y, fmt="%d", delimiter=",")


if __name__ == "__main__":
    save_ground_truth(1000)
    save_ground_truth(3000)
    save_ground_truth(5000)