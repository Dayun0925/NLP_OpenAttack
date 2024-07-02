import torch

from NewsSampler import prepare_query
from file_utils import read_data_from_csv
import numpy as np

if __name__ == '__main__':

    #
    _, labels = prepare_query(seed=0, size=1000)
    _, test_y = prepare_query(seed=0, size=1000)
    
    print("sample labels (first 10):\n%s" % str(labels[:10]))
    # print(out_puts)
    nlp_combined_3models = read_data_from_csv('CNN_test_outputs/3_models_combine_results.csv')
    predictions_tensor = torch.tensor(nlp_combined_3models)
    predicted_labels = torch.argmax(predictions_tensor, dim=1)
    predicted_labels = np.array(predicted_labels)
    correct_predictions = sum(1 for pred, true in zip(predicted_labels, labels) if pred == true)
    print(correct_predictions)
    accuracy = (correct_predictions / len(predicted_labels))
    print('combined 3个模型的精确度是：{}'.format(accuracy))

    nlp_combined_5models = read_data_from_csv('CNN_test_outputs/5_models_combine_results.csv')
    predictions_tensor = torch.tensor(nlp_combined_5models)
    predicted_labels = torch.argmax(predictions_tensor, dim=1)
    predicted_labels = np.array(predicted_labels)
    correct_predictions = sum(1 for pred, true in zip(predicted_labels, labels) if pred == true)
    print(correct_predictions)
    accuracy = (correct_predictions / len(predicted_labels))
    print('combined 5个模型的精确度是：{}'.format(accuracy))

    nlp_combined_7models = read_data_from_csv('CNN_test_outputs/7_models_combine_results.csv')
    predictions_tensor = torch.tensor(nlp_combined_7models)
    predicted_labels = torch.argmax(predictions_tensor, dim=1)
    predicted_labels = np.array(predicted_labels)
    correct_predictions = sum(1 for pred, true in zip(predicted_labels, labels) if pred == true)
    print(correct_predictions)
    accuracy = (correct_predictions / len(predicted_labels))
    print('combined 7个模型的精确度是：{}'.format(accuracy))