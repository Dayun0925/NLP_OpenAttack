import pickle
import tensorflow as tf
import numpy as np
import OpenAttack
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.layers import TextVectorization
import math
import time

model_num = 5  # 更新模型数量
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
    outputs = np.zeros((rows, columns))
    for j in range(rows):
        temp_output = np.zeros(label_num)
        denominator = np.sum(model_inputs.weight)
        for i in range(model_num):
            temp_output += model_inputs.weight[i] * model_inputs.preds[i][j]
        outputs[j] = temp_output / denominator
    return outputs


def get_L2_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def softmax(input_weights):
    exp_weights = np.exp(input_weights)
    return exp_weights / np.sum(exp_weights)


def update_weights(model_inputs, outputs):
    query_size = model_inputs.query_size
    old_weights = model_inputs.weight.copy()
    distances = np.zeros((model_num, query_size))
    numerators = np.zeros(model_num)
    for i in range(model_num):
        for j in range(query_size):
            distances[i][j] = get_L2_distance(model_inputs.preds[i][j], outputs[j])
        numerators[i] = np.sum(distances[i])
    denominator = np.sum(numerators)
    for i in range(model_num):
        if numerators[i] == 0:
            model_inputs.weight[i] = old_weights[i]
        else:
            model_inputs.weight[i] = -math.log(numerators[i] / denominator)
    return softmax(model_inputs.weight)


def truth_discovery(model_inputs):
    for _ in range(iteration_num):
        outputs = calculate_current_outputs(model_inputs)
        model_inputs.weight = update_weights(model_inputs, outputs)
    return outputs


# 载入数据
newsgroups_test = fetch_20newsgroups(subset='test')
X_test = newsgroups_test.data[:100]
y_test = newsgroups_test.target[:100]

vectorizer = TextVectorization(max_tokens=75000, output_sequence_length=500)
vectorizer.adapt(X_test)
X_test_vectorized = vectorizer(X_test).numpy()

# 定义 TruthDiscoveryClassifier 类
class TruthDiscoveryClassifier(OpenAttack.Classifier):
    def __init__(self, truth_discovery_func):
        self.truth_discovery_func = truth_discovery_func
        self.CNN_model = tf.keras.models.load_model(r'../trained_models/20News_CNN.keras')
        with open("../trained_models/20News_boost.pkl", "rb") as f:
            self.boost_model = pickle.load(f)
        with open("../trained_models/20News_DecisionTree.pkl", "rb") as f:
            self.decisionTree_model = pickle.load(f)
        self.RNN_model = tf.keras.models.load_model(r'../trained_models/20News_RNN.keras')  # 载入 RNN 模型
        with open("../trained_models/20News_bagging.pkl", "rb") as f:
            self.bagging_model = pickle.load(f)  # 载入 Bagging 模型

    def get_pred(self, input_):
        return np.array(self.get_prob(input_)).argmax(axis=1)

    def get_prob(self, input_):
        input_vectorized = vectorizer(input_).numpy()
        cnn_preds = self.CNN_model.predict(input_vectorized, batch_size=32)
        rnn_preds = self.RNN_model.predict(input_vectorized, batch_size=32)
        boost_preds = self.boost_model.predict_proba(input_)
        decisionTree_preds = self.decisionTree_model.predict_proba(input_)
        bagging_preds = self.bagging_model.predict_proba(input_)

        models_preds = np.array([cnn_preds, boost_preds, decisionTree_preds, rnn_preds, bagging_preds])
        model_inputs = ModelInput(models_preds, np.ones(model_num) * 1, len(input_))
        combined_predictions = self.truth_discovery_func(model_inputs)
        return np.array(combined_predictions)


victim = TruthDiscoveryClassifier(truth_discovery)
attacker = OpenAttack.attackers.DeepWordBugAttacker()

test_data_subset = X_test[:100]
dataset = [{"x": x, "y": y} for x, y in zip(test_data_subset, y_test[:100])]

start_time = time.time()
attack_eval = OpenAttack.AttackEval(attacker, victim)
attack_eval.eval(dataset, visualize=True)
end_time = time.time()

print("Total running time: ", end_time - start_time)
