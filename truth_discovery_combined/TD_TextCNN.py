import argparse
import math
import os.path

from file_utils import read_data_from_csv, write_outputs_to_csv
import numpy as np

iteration_num = 100
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

    parser = argparse.ArgumentParser(description="使用truth_discovery算法组合多个CNN模型.")
    parser.add_argument("-model_num", type=str, required=True,
                        help="输入模型的数量，有三个数量可供选择(3,5,7)")
    parser.add_argument("-combine_type", type=str, required=True,
                        help="对测试数据集或者训练数据集的结果进行combine.，输入为:test 或者 train")

    args = parser.parse_args()
    # 解析输入参数
    model_num = int(args.model_num)
    combine_type = args.combine_type
    # 获取预测结果 不添加噪声
    # 测试数据集的预测结果

    # 定义存放多个模型预测结果的Numpy数组，数组维度为（模型数量，查询数量=1000，标签数量=20）
    # 测试集的预测结果
    CNN_1_test = read_data_from_csv("../20news_predict/outputs/CNN_7_models/CNN_1_model.csv")
    CNN_2_test = read_data_from_csv("../20news_predict/outputs/CNN_7_models/CNN_2_model.csv")
    CNN_3_test = read_data_from_csv("../20news_predict/outputs/CNN_7_models/CNN_3_model.csv")
    CNN_4_test = read_data_from_csv("../20news_predict/outputs/CNN_7_models/CNN_4_model.csv")
    CNN_5_test = read_data_from_csv("../20news_predict/outputs/CNN_7_models/CNN_5_model.csv")
    CNN_6_test = read_data_from_csv("../20news_predict/outputs/CNN_7_models/CNN_6_model.csv")
    CNN_7_test = read_data_from_csv("../20news_predict/outputs/CNN_7_models/CNN_7_model.csv")
    # 训练集的预测结果
    CNN_1_train = read_data_from_csv("../20news_predict/outputs/CNN_7_models_train/CNN_1_model.csv")
    CNN_2_train = read_data_from_csv("../20news_predict/outputs/CNN_7_models_train/CNN_2_model.csv")
    CNN_3_train = read_data_from_csv("../20news_predict/outputs/CNN_7_models_train/CNN_3_model.csv")
    CNN_4_train = read_data_from_csv("../20news_predict/outputs/CNN_7_models_train/CNN_4_model.csv")
    CNN_5_train = read_data_from_csv("../20news_predict/outputs/CNN_7_models_train/CNN_5_model.csv")
    CNN_6_train = read_data_from_csv("../20news_predict/outputs/CNN_7_models_train/CNN_6_model.csv")
    CNN_7_train = read_data_from_csv("../20news_predict/outputs/CNN_7_models_train/CNN_7_model.csv")

    x = model_num
    y = query_size
    z = label_num
    prediction_ouputs = np.array((x, y, z))
    if combine_type == "test":
        if model_num == 3:
            prediction_ouputs=[CNN_1_test, CNN_3_test, CNN_5_test]
        elif model_num == 5:
            prediction_ouputs = [CNN_1_test, CNN_3_test, CNN_5_test, CNN_6_test, CNN_7_test]
        elif model_num == 7:
            prediction_ouputs = [CNN_1_test, CNN_3_test, CNN_5_test, CNN_6_test, CNN_7_test, CNN_2_test, CNN_4_test]
        else:
            raise Exception('请输入指定的模型数量，"3"、"5"或者"7"')
    elif combine_type == 'train':
        if model_num == 3:
            prediction_ouputs = [CNN_1_train, CNN_3_train, CNN_5_train]
        elif model_num == 5:
            prediction_ouputs = [CNN_1_train, CNN_3_train, CNN_5_train, CNN_6_train, CNN_7_train]
        elif model_num == 7:
            prediction_ouputs = [CNN_1_train, CNN_3_train, CNN_5_train, CNN_6_train, CNN_7_train,CNN_2_train, CNN_4_train]
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
        output_path = os.path.join(os.getcwd(), "CNN_test_outputs",
                                   str(model_num) + "_models_combine_results.csv")
    else:
        output_path = os.path.join(os.getcwd(), "CNN_train_outputs",
                                   str(model_num) + "_models_combine_results.csv")
    write_outputs_to_csv(output_path=output_path, outputs_result=result_outputs)
