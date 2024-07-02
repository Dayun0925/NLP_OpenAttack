import argparse
import math
import os.path

from file_utils import read_data_from_csv, write_outputs_to_csv
import numpy as np


iteration_num = 20
query_size = 1000
label_num = 20


# 定义模型输入的结果
class ModelInput:
    def __init__(self, preds, weight, model_num):
        # 各个模型预测结果数组
        self.preds = preds
        # 各个模型的权重
        self.weight = weight
        self.model_num = model_num


def calculate_current_outputs(model_inputs):
    rows = query_size
    columns = label_num
    initial_value = 0
    model_num = model_inputs.model_num
    outputs = [[initial_value for _ in range(columns)] for _ in range(rows)]

    for j in range(query_size):
        denominator = 0.0
        temp_output = [0] * label_num
        for i in range(model_num):
            denominator += model_inputs.weight[i]
        for i in range(model_num):
            for k in range(label_num):
                temp_output[k] += model_inputs.weight[i] * model_inputs.preds[i][j][k]
            for k in range(label_num):
                outputs[j][k] = temp_output[k] / denominator
    return outputs


def softmax_2(output, length):
    denominator = 0.0
    for i in range(length):
        denominator += math.exp(output[i])
    for i in range(length):
        output[i] = math.exp(output[i]) / denominator


def get_L2_distance(vec1, vec2, size):
    distance = 0.0
    for i in range(size):
        distance += math.pow(vec1[i] - vec2[i], 2)
    distance = math.sqrt(distance)
    return distance


def softmax(input, length):
    denominator = 0.0
    for i in range(length):
        denominator += math.exp(input.weight[i])
    for i in range(length):
        input.weight[i] = math.exp(input.weight[i]) / denominator
    return input.weight


def update_weights(model_inputs, outputs):
    model_num = model_inputs.model_num
    old_weights = [0] * model_num

    for i in range(model_num):
        old_weights[i] = model_inputs.weight[i]

    numerators = [0] * model_num
    denominator = 0.0
    rows = model_num
    columns = query_size
    initial_value = 0
    distances = [[initial_value for _ in range(columns)] for _ in range(rows)]
    for i in range(model_num):
        for j in range(query_size):
            distances[i][j] = get_L2_distance(model_inputs.preds[i][j], outputs[j], label_num)
            numerators[i] += distances[i][j]
        denominator += numerators[i]
    # 更新权重
    for i in range(model_num):
        temp = numerators[i] / denominator
        if math.isnan(temp) or temp == 0:
            model_inputs.weight[i] = old_weights[i]
        else:
            model_inputs.weight[i] = -math.log(temp)
    weight = softmax(model_inputs, model_num)
    return weight


def truth_discovery(model_inputs):
    r = 0
    while r < iteration_num:
        outputs = calculate_current_outputs(model_inputs)
        # for j in range(query_size):
        #     softmax_2(outputs[j], label_num)
        model_inputs.weight = update_weights(model_inputs, outputs)
        r += 1
    return outputs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="使用truth_discovery算法组合多个模型.")
    parser.add_argument("-model_num", type=str, required=True,
                        help="输入模型的数量，有三个数量可供选择(3,5,7)")
    parser.add_argument("-combine_type", type=str, required=True,
                        help="对测试数据集或者训练数据集的结果进行combine.，输入为:test 或者 train")
    parser.add_argument("-add_noise", type=bool, required=True,
                        help="是否添加噪声，True 或者 False")
    args = parser.parse_args()
    # 解析输入参数
    model_num = int(args.model_num)
    combine_type = args.combine_type
    add_noise =args.add_noise
    # 获取预测结果 不添加噪声
    # 测试数据集的预测结果
    svm_test = read_data_from_csv("../predict_outputs/pre_outputs/svm.csv")
    knn_test = read_data_from_csv("../predict_outputs/pre_outputs/knn.csv")
    cnn_test = read_data_from_csv("../predict_outputs/pre_outputs/CNN.csv")
    RF_test = read_data_from_csv("../predict_outputs/pre_outputs/srandom_forest.csv")
    DT_test = read_data_from_csv("../predict_outputs/pre_outputs/decision_tree.csv")
    rnn_test = read_data_from_csv("../predict_outputs/pre_outputs/RNN.csv")
    rcnn_test = read_data_from_csv("../predict_outputs/pre_outputs/RCNN.csv")

    svm_train = read_data_from_csv("../predict_outputs/train_outputs/svm_train.csv")
    knn_train = read_data_from_csv("../predict_outputs/train_outputs/knn_train.csv")
    cnn_train = read_data_from_csv("../predict_outputs/train_outputs/CNN_train.csv")
    RF_train = read_data_from_csv("../predict_outputs/train_outputs/srandom_forest_train.csv")
    DT_train = read_data_from_csv("../predict_outputs/train_outputs/decision_tree_train.csv")
    rnn_train = read_data_from_csv("../predict_outputs/train_outputs/RNN_train.csv")
    rcnn_train = read_data_from_csv("../predict_outputs/train_outputs/RCNN_train.csv")
    # 定义存放多个模型预测结果的Numpy数组，数组维度为（模型数量，查询数量=1000，标签数量=20）

    # 添加噪声的预测结果
    svm_noise_test = read_data_from_csv("../add_noise/noise_prediction/test/svm_noisy_result.csv")
    knn_noise_test = read_data_from_csv("../add_noise/noise_prediction/test/knn_noisy_result.csv")
    cnn_noise_test = read_data_from_csv("../add_noise/noise_prediction/test/CNN_noisy_result.csv")
    RF_noise_test = read_data_from_csv("../add_noise/noise_prediction/test/srandom_noisy_result.csv")
    DT_noise_test = read_data_from_csv("../add_noise/noise_prediction/test/decision_noisy_result.csv")
    rnn_noise_test = read_data_from_csv("../add_noise/noise_prediction/test/RNN_noisy_result.csv")
    rcnn_noise_test = read_data_from_csv("../add_noise/noise_prediction/test/RCNN_noisy_result.csv")

    svm_noise_train = read_data_from_csv("../add_noise/noise_prediction/train/svm_noisy_result.csv")
    knn_noise_train = read_data_from_csv("../add_noise/noise_prediction/train/knn_noisy_result.csv")
    cnn_noise_train = read_data_from_csv("../add_noise/noise_prediction/train/CNN_noisy_result.csv")
    RF_noise_train = read_data_from_csv("../add_noise/noise_prediction/train/srandom_noisy_result.csv")
    DT_noise_train = read_data_from_csv("../add_noise/noise_prediction/train/decision_noisy_result.csv")
    rnn_noise_train = read_data_from_csv("../add_noise/noise_prediction/train/RNN_noisy_result.csv")
    rcnn_noise_train = read_data_from_csv("../add_noise/noise_prediction/train/RCNN_noisy_result.csv")

    x = model_num
    y = query_size
    z = label_num
    prediction_ouputs = np.array((x, y, z))
    if combine_type == "test":
        if model_num == 3:
            if add_noise == True:
                prediction_ouputs=[svm_noise_test, knn_noise_test, cnn_noise_test]
            elif add_noise ==False:
                prediction_ouputs = [svm_test, knn_test, cnn_test]
            else:
                raise Exception('请输入是否添加噪声 ，-add_noise=True或-add_noise=False')
        elif model_num == 5:
            if add_noise == True:
                prediction_ouputs=[svm_noise_test, knn_noise_test, cnn_noise_test, RF_noise_test, DT_noise_test]
            elif add_noise ==False:
                prediction_ouputs = [svm_test, knn_test, cnn_test, RF_test, DT_test]
            else:
                raise Exception('请输入是否添加噪声 ，-add_noise=True或-add_noise=False')

        elif model_num == 7:
            if add_noise == True:
                prediction_ouputs=[svm_noise_test, knn_noise_test, cnn_noise_test, RF_noise_test, DT_noise_test, rnn_noise_test, rcnn_noise_test]
            elif add_noise == False:
                prediction_ouputs = [svm_test, knn_test, cnn_test, RF_test, DT_test, rnn_test, rcnn_test]
            else:
                raise Exception('请输入是否添加噪声 ，-add_noise=True或-add_noise=False')


        else:
            raise Exception('请输入指定的模型数量，"3"、"5"或者"7"')
    elif combine_type == 'train':
        if model_num == 3:
            if add_noise == True:
                prediction_ouputs=[svm_noise_train, knn_noise_train, cnn_noise_train]
            elif add_noise == False:
                prediction_ouputs = [svm_train, knn_train, cnn_train]
            else:
                raise Exception('请输入是否添加噪声 ，-add_noise=True或-add_noise=False')

        elif model_num == 5:
            if add_noise == True:
                prediction_ouputs=[svm_noise_train, knn_noise_train, cnn_noise_train,RF_noise_train,DT_noise_train]
            elif add_noise == False:
                prediction_ouputs = [svm_train, knn_train, rnn_train, RF_train, DT_train]
            else:
                raise Exception('请输入是否添加噪声 ，-add_noise=True或-add_noise=False')

        elif model_num == 7:
            if add_noise == True:
                prediction_ouputs=[svm_noise_train, knn_noise_train, cnn_noise_train,RF_noise_train,DT_noise_train,rnn_noise_train,rcnn_noise_train]
            elif add_noise == False:
                prediction_ouputs = [svm_train, knn_train, rnn_train, RF_train, DT_train, rnn_train, rcnn_train]
            else:
                raise Exception('请输入是否添加噪声 ，-add_noise=True或-add_noise=False')

        else:
            raise Exception('请输入指定的模型数量，"3"、"5"或者"7"')
    else:
        raise Exception('请输入指定的数据集预测结果,"test"或者"train"')

    weights = np.ones(model_num) * 1
    model_inputs = ModelInput(prediction_ouputs, weights, model_num)

    # 使用truth discovery算法进行输出结果
    out_puts = truth_discovery(model_inputs)
    result_outputs = np.array(out_puts)
    if combine_type == 'test':
        if add_noise == False:
            output_path = os.path.join(os.getcwd(), "combine_test_outputs",
                                       str(model_num) + "_models_combine_results.csv")
        elif add_noise==True:
            output_path = os.path.join(os.getcwd(), "combine_noise_test_outputs",
                                       str(model_num) + "_models_combine_results.csv")
        else:
            raise Exception('请输入是否添加噪声 ，-add_noise=True或-add_noise=False')
    else:
        if add_noise == False:
            output_path = os.path.join(os.getcwd(), "combine_train_outputs",
                                       str(model_num) + "_models_combine_results.csv")
        elif add_noise ==True:
            output_path = os.path.join(os.getcwd(), "combine_noise_train_outputs",
                                       str(model_num) + "_models_combine_results.csv")
        else:
            raise Exception('请输入是否添加噪声 ，-add_noise=True或-add_noise=False')

    write_outputs_to_csv(output_path=output_path, outputs_result=result_outputs)
