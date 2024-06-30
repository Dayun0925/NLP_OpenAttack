import os
import numpy as np
import pandas as pd
from type_utils import *
from file_utils import read_data_from_csv, write_outputs_to_csv


def get_data(data_path):
    currentPath = os.getcwd()
    data = {}
    file_names = os.listdir(data_path)
    for file in file_names:
        if '.ipynb_checkpoints' in file:
            continue
        if data_path == "prediction":
            model_name = file.split('.')[0]
        else:
            model_name = file.split('_')[0]

        file_path = os.path.join(currentPath, data_path, file)
        print(f"Processing file: {file_path}")

        try:
            data[model_name] = read_data_from_csv(file_path)
        except ValueError as e:
            print(f"Error reading file {file_path}: {e}")
            raise

    return data


if __name__ == "__main__":
    t20News_category_num = 20

    # 添加噪声到所有预测结果
    prediction_path = "../predict_outputs/pre_outputs"
    noise_path = "outputs/20News/query_1000/type_probs"  # 修改为正确的噪声结果路径
    model_noise = get_data(noise_path)
    model_prediction = get_data(prediction_path)
    prediction_add_noise = np.array((1000, 20), dtype=np.float64)

    # 新建 noise_prediction 文件夹
    output_dir = os.path.join(os.getcwd(), "noise_prediction")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key in model_prediction.keys():
        model_name = key.split('.')[0]
        prediction = model_prediction[key]
        noise = model_noise[model_name]
        noise_prediction = prediction + noise*0.1
        write_path = os.path.join(output_dir, model_name + "_noisy_result.csv")
        write_outputs_to_csv(output_path=write_path, outputs_result=noise_prediction)
