import torch
import numpy as np

def read_data_from_csv(file_path):
    data = np.loadtxt(file_path, delimiter=',')  # 根据需要调整此行代码以匹配文件格式
    return data

def calculate_accuracy(predictions, labels):
    predicted_labels = torch.argmax(predictions, dim=1).numpy()
    correct_predictions = np.sum(predicted_labels == labels)
    accuracy = correct_predictions / len(labels)
    return accuracy

# 读取数据
nlp_combined_3models = read_data_from_csv('../20news_predict/outputs/CNN_7_models_train/CNN_1_model.csv')

# 将预测结果转换为 tensor
predictions_tensor = torch.tensor(nlp_combined_3models)

# 假设 labels 是您实际的标签数据
labels = np.array([...])  # 请用实际标签数据替换

# 打印一些预测结果和标签，确保它们合理
print("Predictions:", predictions_tensor[:10])
print("Predicted Labels:", torch.argmax(predictions_tensor, dim=1).numpy()[:10])
print("True Labels:", labels[:10])

# 计算精确度
accuracy = calculate_accuracy(predictions_tensor, labels)

print(f'combined 3个模型的精确度是：{accuracy}')
