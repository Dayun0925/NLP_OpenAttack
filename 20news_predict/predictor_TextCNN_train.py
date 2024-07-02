import sys
import random
import os
import pickle
import tensorflow.keras
import numpy as np
from keras.utils import pad_sequences  # Updated import

from NewsSampler_train_data import *

from type_utils import *

import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CATEGORY_NUM = 20


def data_preprocess_Tokenizer(X, MAX_NB_WORDS=75000, MAX_SEQUENCE_LENGTH=500):
    ## TODO: need test
    text = X.copy()
    tokenizer = pickle.load(open("../tokenizer/CNN_tokenizer.pkl", "rb"))
    sequences = tokenizer.texts_to_sequences(text)
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    indices = np.arange(text.shape[0])
    text = text[indices]
    X_processed = text
    return X_processed


def load_model(model_path):
    model = None
    _, model_suffix = os.path.splitext(model_path)
    print("loading model %s" % model_path)
    if model_suffix in ['.keras']:
        print("load keras models")
        model = tensorflow.keras.models.load_model(model_path)
    elif model_suffix in ['.pkl']:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    elif model_suffix in ['.pt']:
        # model = torch.load(model_path)
        pass
    return model


def do_test_keras_CNN(query_size, model_path):
    model = load_model(model_path)
    test_X, _ = prepare_query(seed=0, size=query_size)
    test_X = data_preprocess_Tokenizer(test_X)
    pred_y = model.predict(test_X)
    return pred_y


def collect_in_ones(q_size):
    save_path_probs = "./outputs/CNN_7_models_train/"
    CNN_dir = "../trained_models/CNN_models"
    files = os.listdir(CNN_dir)
    index = 1

    if not os.path.exists(save_path_probs):
        os.makedirs(save_path_probs)

    for file in files:
        model_path = os.path.join(CNN_dir, file)
        preds = do_test_keras_CNN(query_size=q_size, model_path=model_path)
        preds_probs = convert_output_type(preds, OUTPUT_TYPE.TYPE_PROBS)
        save_results_to_txt(preds_probs, "%s/CNN_%s_model.csv" % (save_path_probs, index))
        index = index + 1


if __name__ == "__main__":
    q_sizes = [1000]
    for q_size in q_sizes:
        collect_in_ones(q_size)
