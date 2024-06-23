import pickle
import tensorflow as tf
import numpy as np
import OpenAttack
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.layers import TextVectorization
import math

# 定义 truth_discovery 算法及相关函数
model_num = 3
iteration_num = 20
label_num = 20


class ModelInput:
    def __init__(self, preds, weight, query_size):
        self.preds = preds
        self.weight = weight
        self.query_size = query_size


def calculate_current_outputs(model_inputs):

    rows = model_inputs.query_size
    columns = label_num
    initial_value = 0
    outputs = [[initial_value for _ in range(columns)] for _ in range(rows)]
    for j in range(rows):
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
    query_size = model_inputs.query_size
    old_weights = [0] * model_num
    for i in range(model_num):
        old_weights[i] = model_inputs.weight[i]
    numerators = [0] * model_num
    denominator = 0.0
    distances = np.zeros((model_num, query_size))
    for i in range(model_num):
        for j in range(query_size):
            distances[i][j] = get_L2_distance(model_inputs.preds[i][j], outputs[j], label_num)
            numerators[i] += distances[i][j]
        denominator += numerators[i]
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
        model_inputs.weight = update_weights(model_inputs, outputs)
        r += 1
    return outputs


# 载入数据集
newsgroups_test = fetch_20newsgroups(subset='test')
X_test = newsgroups_test.data
y_test = newsgroups_test.target


class TruthDiscoveryClassifier(OpenAttack.Classifier):
    def __init__(self, truth_discovery_func):
        self.truth_discovery_func = truth_discovery_func

    def get_pred(self, input_):
        return np.array(self.get_prob(input_)).argmax(axis=1)

    def get_prob(self, input_):
        # 载入不同的模型
        CNN_model = tf.keras.models.load_model(r'trained_models/20News_CNN.keras')
        with open("trained_models/20news_boost.pkl", "rb") as f:
            boost_model = pickle.load(f)
        with open("trained_models/20News_DecisionTree.pkl", "rb") as f:
            decisionTree_model = pickle.load(f)

        # 创建并适配 TextVectorization 层
        max_sequence_length = 500
        vectorizer = TextVectorization(max_tokens=75000, output_sequence_length=max_sequence_length)
        vectorizer.adapt(input_)
        X_test_vectorized = vectorizer(input_).numpy()

        # 使用模型对测试数据进行预测
        cnn_preds = CNN_model.predict(X_test_vectorized)

        # 对于 scikit-learn 模型，需要使用原始文本数据进行预测
        boost_preds = boost_model.predict_proba(input_)
        decisionTree_preds = decisionTree_model.predict_proba(input_)
        # 将预测结果转换为合适的格式
        models_preds = np.zeros((model_num, len(input_), label_num))
        models_preds[0] = cnn_preds
        models_preds[1] = boost_preds
        models_preds[2] = decisionTree_preds
        model_inputs = ModelInput(models_preds, np.ones(model_num)*1, len(input_))
        combined_predictions = self.truth_discovery_func(model_inputs)
        return np.array(combined_predictions)  # Ensure this returns a NumPy array


# 优化后运行评估
victim = TruthDiscoveryClassifier(truth_discovery)

attacker = OpenAttack.attackers.DeepWordBugAttacker()
# 使用更小的数据集进行快速测试
test_data_subset = X_test[:100]
dataset = [{"x": x, "y": y} for x, y in zip(test_data_subset, y_test[:100])]

attack_eval = OpenAttack.AttackEval(attacker, victim)
attack_eval.eval(dataset, visualize=True)
