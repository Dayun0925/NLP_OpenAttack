import torch

from NewsSampler import prepare_query
from file_utils import read_data_from_csv
import numpy as np


if __name__ == '__main__':
    bagging_outs = read_data_from_csv('noise_prediction/bagging_noisy_result.csv')
    svm_outs = read_data_from_csv('noise_prediction/svm_noisy_result.csv')
    rnn_outputs = read_data_from_csv('noise_prediction/RNN_noisy_result.csv')
    cnn_outputs = read_data_from_csv('noise_prediction/CNN_noisy_result.csv')
    boost_outputs = read_data_from_csv('noise_prediction/boost_noisy_result.csv')
    knn_outputs = read_data_from_csv('noise_prediction/knn_noisy_result.csv')
    decision_tree_outputs = read_data_from_csv('noise_prediction/decision_noisy_result.csv')
    dnn_outputs = read_data_from_csv('noise_prediction/DNN_noisy_result.csv')
    random_forest_outputs = read_data_from_csv('noise_prediction/srandom_noisy_result.csv')
    rcnn_outputs = read_data_from_csv('noise_prediction/RCNN_noisy_result.csv')
    #
    _, labels = prepare_query(seed=0, size=1000)
    # print(out_puts)
    # 转换成PyTorch的Tensor
    svm_predictions_tensor = torch.tensor(svm_outs)
    svm_predicted_labels = torch.argmax(svm_predictions_tensor, dim=1)
    svm_predicted_labels = np.array(svm_predicted_labels)
    svm_correct_predictions = sum(1 for pred, true in zip(svm_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (svm_correct_predictions / len(svm_predicted_labels)) * 100.0
    print('svm的精确度是：{}'.format(accuracy))
    
    bagging_predictions_tensor = torch.tensor(bagging_outs)
    bagging_predicted_labels = torch.argmax(bagging_predictions_tensor, dim=1)
    bagging_predicted_labels = np.array(bagging_predicted_labels)
    bagging_correct_predictions = sum(1 for pred, true in zip(bagging_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (bagging_correct_predictions / len(bagging_predicted_labels)) * 100.0
    print('bagging的精确度是：{}'.format(accuracy))
    
    rnn_predictions_tensor = torch.tensor(rnn_outputs)
    rnn_predicted_labels = torch.argmax(rnn_predictions_tensor, dim=1)
    rnn_predicted_labels = np.array(rnn_predicted_labels)
    rnn_correct_predictions = sum(1 for pred, true in zip(rnn_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (rnn_correct_predictions / len(rnn_predicted_labels)) * 100.0
    print('rnn的精确度是：{}'.format(accuracy))
    
    cnn_predictions_tensor = torch.tensor(cnn_outputs)
    cnn_predicted_labels = torch.argmax(cnn_predictions_tensor, dim=1)
    cnn_predicted_labels = np.array(cnn_predicted_labels)
    cnn_correct_predictions = sum(1 for pred, true in zip(cnn_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (cnn_correct_predictions / len(cnn_predicted_labels)) * 100.0
    print('cnn的精确度是：{}'.format(accuracy))
    
    boost_predictions_tensor = torch.tensor(boost_outputs)
    boost_predicted_labels = torch.argmax(boost_predictions_tensor, dim=1)
    boost_predicted_labels = np.array(boost_predicted_labels)
    boost_correct_predictions = sum(1 for pred, true in zip(boost_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (boost_correct_predictions / len(boost_predicted_labels)) * 100.0
    print('boost的精确度是：{}'.format(accuracy))
    
    knn_predictions_tensor = torch.tensor(knn_outputs)
    knn_predicted_labels = torch.argmax(knn_predictions_tensor, dim=1)
    knn_predicted_labels = np.array(knn_predicted_labels)
    knn_correct_predictions = sum(1 for pred, true in zip(knn_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (knn_correct_predictions / len(knn_predicted_labels)) * 100.0
    print('knn的精确度是：{}'.format(accuracy))
    
    decision_tree_predictions_tensor = torch.tensor(decision_tree_outputs)
    decision_tree_predicted_labels = torch.argmax(decision_tree_predictions_tensor, dim=1)
    decision_tree_predicted_labels = np.array(decision_tree_predicted_labels)
    decision_tree_correct_predictions = sum(1 for pred, true in zip(decision_tree_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (decision_tree_correct_predictions / len(decision_tree_predicted_labels)) * 100.0
    print('decision_tree的精确度是：{}'.format(accuracy))
    
    dnn_predictions_tensor = torch.tensor(dnn_outputs)
    dnn_predicted_labels = torch.argmax(dnn_predictions_tensor, dim=1)
    dnn_predicted_labels = np.array(dnn_predicted_labels)
    dnn_correct_predictions = sum(1 for pred, true in zip(dnn_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (dnn_correct_predictions / len(dnn_predicted_labels)) * 100.0
    print('dnn的精确度是：{}'.format(accuracy))
    
    random_forest_predictions_tensor = torch.tensor(random_forest_outputs)
    random_forest_predicted_labels = torch.argmax(random_forest_predictions_tensor, dim=1)
    random_forest_predicted_labels = np.array(random_forest_predicted_labels)
    random_forest_correct_predictions = sum(1 for pred, true in zip(random_forest_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (random_forest_correct_predictions / len(random_forest_predicted_labels)) * 100.0
    print('random_forest的精确度是：{}'.format(accuracy))
    
    rcnn_predictions_tensor = torch.tensor(rcnn_outputs)
    rcnn_predicted_labels = torch.argmax(rcnn_predictions_tensor, dim=1)
    rcnn_predicted_labels = np.array(rcnn_predicted_labels)
    rcnn_correct_predictions = sum(1 for pred, true in zip(rcnn_predicted_labels, labels) if pred == true)
    # print(correct_predictions)
    accuracy = (rcnn_correct_predictions / len(rcnn_predicted_labels)) * 100.0
    print('rcnn的精确度是：{}'.format(accuracy))
    
    
#     nlp_combined = read_data_from_csv('./results/TD_nlp/nlp_boost_cnn_DT__rnn_bagging__randomforest_svm.csv')
#     predictions_tensor = torch.tensor(nlp_combined)
#     predicted_labels = torch.argmax(predictions_tensor, dim=1)
#     predicted_labels = np.array(predicted_labels)
#     correct_predictions = sum(1 for pred, true in zip(predicted_labels, labels) if pred == true)
#     # print(correct_predictions)
#     accuracy = (correct_predictions / len(predicted_labels))
#     print('combined的精确度是：{}'.format(accuracy))