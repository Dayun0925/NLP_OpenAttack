from keras.layers import Dropout, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
import numpy as np
import os

# 创建一个内存映射文件以存储TF-IDF特征
def create_memmap(filename, shape, dtype='float32'):
    return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

# 将TF-IDF特征保存到内存映射文件
def save_tfidf_to_memmap(data, vectorizer, filename):
    memmap = create_memmap(filename, (len(data), vectorizer.max_features))
    for i, text in enumerate(data):
        memmap[i] = vectorizer.transform([text]).toarray()
    return memmap

# 定义TF-IDF转换器
def TFIDF(X_train, X_test, MAX_NB_WORDS=5000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    vectorizer_x.fit(X_train)
    print("TF-IDF with", MAX_NB_WORDS, "features")
    return vectorizer_x

# 构建DNN模型
def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
    model = Sequential()
    node = 512  # number of nodes
    nLayers = 4  # number of hidden layers

    model.add(Dense(node, input_dim=shape, activation='relu'))
    model.add(Dropout(dropout))
    for _ in range(nLayers):
        model.add(Dense(node, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# 获取数据
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# 初始化TF-IDF转换器
vectorizer_x = TFIDF(X_train, X_test)

# 保存TF-IDF特征到内存映射文件
train_filename = 'X_train_tfidf.dat'
test_filename = 'X_test_tfidf.dat'

X_train_tfidf = save_tfidf_to_memmap(X_train, vectorizer_x, train_filename)
X_test_tfidf = save_tfidf_to_memmap(X_test, vectorizer_x, test_filename)

# 释放内存
del X_train
del X_test

# 构建模型
model_DNN = Build_Model_DNN_Text(vectorizer_x.max_features, 20)
model_DNN.summary()

# 设置批量大小和训练轮数
batch_size = 128
epochs = 20

# 计算每个轮次的步数和验证步数
steps_per_epoch = len(X_train_tfidf) // batch_size
validation_steps = len(X_test_tfidf) // batch_size

# 训练模型
model_DNN.fit(
    X_train_tfidf, y_train,
    validation_data=(X_test_tfidf, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2
)

# 获取预测结果
predicted = model_DNN.predict(X_test_tfidf)
predicted_classes = np.argmax(predicted, axis=1)

# 打印分类报告
print(metrics.classification_report(y_test, predicted_classes))

# 保存模型
model_DNN.save('../trained_models/20News_DNN.keras')
