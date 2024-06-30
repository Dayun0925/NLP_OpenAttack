import os
import pickle
import ssl
import sys
import re
import argparse
import tensorflow as tf
import OpenAttack
import keras
from OpenAttack.attackers import DeepWordBugAttacker, TextBuggerAttacker, PWWSAttacker, GeneticAttacker
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.layers import TextVectorization

# 确保SSL上下文
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Python编码
os.environ['PYTHONIOENCODING'] = 'UTF-8'
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 载入数据集
newsgroups_test = fetch_20newsgroups(subset='test')
X_test = newsgroups_test.data
y_test = newsgroups_test.target

# 加载不同的模型
cnn_model_path = "../trained_models/20News_CNN.keras"
rnn_model_path = "../trained_models/20News_RNN.keras"
dnn_model_path = "../trained_models/20News_DNN.keras"
boost_model_path = "../trained_models/20News_boost.pkl"
decision_tree_model_path = "../trained_models/20News_DecisionTree.pkl"
bagging_model_path = "../trained_models/20News_bagging.pkl"
knn_model_path = "../trained_models/20News_KNN.pkl"
rcnn_model_path = "../trained_models/20News_RCNN.keras"
svm_model_path = "../trained_models/20News_SVM_probs.pkl"
random_forest_model_path = "../trained_models/20News_randomForest.pkl"
models = {}
if os.path.isfile(cnn_model_path):
    models['cnn'] = tf.keras.models.load_model(cnn_model_path)

if os.path.isfile(rcnn_model_path):
    models['rcnn'] = tf.keras.models.load_model(rcnn_model_path)

if os.path.isfile(rnn_model_path):
    print(rnn_model_path)
    models['rnn'] = tf.keras.models.load_model(rnn_model_path)

if os.path.isfile(dnn_model_path):
    models['dnn'] = keras.models.load_model(dnn_model_path)

if os.path.isfile(boost_model_path):
    with open(boost_model_path, 'rb') as f:
        models['boost'] = pickle.load(f)

if os.path.isfile(decision_tree_model_path):
    with open(decision_tree_model_path, 'rb') as f:
        models['decision_tree'] = pickle.load(f)

if os.path.isfile(bagging_model_path):
    with open(bagging_model_path, 'rb') as f:
        models['bagging'] = pickle.load(f)

if os.path.isfile(knn_model_path):
    with open(knn_model_path, 'rb') as f:
        models['knn'] = pickle.load(f)

if os.path.isfile(random_forest_model_path):
    with open(random_forest_model_path, 'rb') as f:
        models['random_forest'] = pickle.load(f)

if os.path.isfile(svm_model_path):
    with open(svm_model_path, 'rb') as f:
        models['svm'] = pickle.load(f)


# 定义自定义分类器类
class CustomClassifier(OpenAttack.Classifier):
    def __init__(self, model, vectorizer=None):
        self.model = model
        self.vectorizer = vectorizer

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        if isinstance(self.model, tf.keras.Model):
            # 适用于TensorFlow模型
            if self.vectorizer:
                input_ = self.vectorizer(input_).numpy()
            predictions = self.model.predict(input_)
        else:
            # 适用于scikit-learn模型
            predictions = self.model.predict_proba(input_)
        return predictions


import unicodedata
import chardet


def preprocess_data(data):
    """
    Preprocess the input data to ensure it conforms to expected format.
    """
    cleaned_data = []
    for item in data:
        text = item['x']

        # 检测并统一文本编码
        encoding = chardet.detect(text.encode())['encoding']
        text = text.encode(encoding, errors='replace').decode('utf-8')

        # 移除控制字符
        control_chars = ''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))
        control_char_re = re.compile(r'[{}]'.format(re.escape(control_chars)))
        text = control_char_re.sub(' ', text)

        # 规范化字符
        text = unicodedata.normalize('NFKD', text)

        # 移除特殊字符
        text = re.sub(r'[^\w\s]', '', text)

        # 转为小写
        text = text.lower()

        # 去除多余空白
        text = ' '.join(text.split())

        # 删除只有下划线或空字符的单词
        text = ' '.join([word for word in text.split() if word])

        cleaned_data.append({'x': text, 'y': item['y']})
    return cleaned_data


def get_vectorizer_for_model(model, X_test):
    # 默认的 output_sequence_length
    output_sequence_length = 500

    if model.input_shape and len(model.input_shape) > 1:
        output_sequence_length = model.input_shape[1]

    vectorizer = TextVectorization(max_tokens=75000, output_sequence_length=output_sequence_length)
    vectorizer.adapt(X_test)
    return vectorizer


# 定义一个函数来选择和评估模型
def evaluate_model(model_name, attacker_type, X_test, y_test):
    # 定义攻击方法
    attack_methods = {
        "DeepWordBug": DeepWordBugAttacker(),
        "TextBugger": TextBuggerAttacker(),
        "PWWS": PWWSAttacker(),
        "Genetic": GeneticAttacker()
    }
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found.")
    if attacker_type not in attack_methods:
        raise ValueError(f"Model '{attacker_type}' not found.")

    attacker = attack_methods[attacker_type]
    model = models[model_name]

    # 创建对应的分类器实例
    if isinstance(model, tf.keras.Model):
        if model_name == 'dnn':
            vectorizer = TextVectorization(max_tokens=75000, output_sequence_length=75000)
            print("为DNN模型设置了 vectorizer 的 output_sequence_length 为 75000")
        else:
            vectorizer = TextVectorization(max_tokens=75000, output_sequence_length=500)
            print(f"为模型 {model_name} 设置了 vectorizer 的 output_sequence_length 为 500")
        vectorizer.adapt(X_test)
        victim = CustomClassifier(model, vectorizer)
    else:
        victim = CustomClassifier(model)

    # 使用OpenAttack进行对抗攻击评估
    attack_eval = OpenAttack.AttackEval(attacker, victim)

    # 评估对抗攻击效果
    dataset = [{"x": x, "y": y} for x, y in zip(X_test[:100], y_test[:100])]
    cleaned_dataset = preprocess_data(dataset)
    attack_eval.eval(cleaned_dataset, visualize=True)


# 设置命令行参数解析
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a text classifier with adversarial attacks.")
    parser.add_argument("-model_type", type=str, required=True,
                        help="The type of model to use (cnn, boost, decision_tree).")
    parser.add_argument("-attacker_type", type=str, required=True,
                        help="The type of model to use (DeepWordBug, TextBugger, PWWS, Genetic).")
    args = parser.parse_args()

    model_type = args.model_type.lower()

    attacker_type = args.attacker_type

    evaluate_model(model_type, attacker_type, X_test, y_test)
