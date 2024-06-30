import csv
import numpy as np


def read_data_from_csv(file_path):
    data = []

    with open(file_path, 'r',encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        # 逐行读取数据
        for row in csv_reader:
            # 将每行数据转换为浮点型并添加到列表
            data.append([float(value) for value in row])
        array_data = np.array(data)

    return array_data


def write_outputs_to_csv(output_path, outputs_result):
    # outputs_result使用numpy数组作为输入
    data = outputs_result.tolist()
    

    # 写入 CSV 文件
    with open(output_path, 'w', newline='') as file:
        # 创建 CSV 写入器
        csv_writer = csv.writer(file)

        # 逐行写入数据
        for row in data:
            csv_writer.writerow(row)


if __name__ == '__main__':
    # 调用函数读取 CSV 文件并打印数据
    csv_data = read_data_from_csv('../image/tasks/MNIST/outputs/MNIST/query_1000/type_probs/MLP.csv')
    # print(csv_data)
    # write_outputs_to_csv('outputs_test.csv',csv_data)
    print("ndim:",csv_data.ndim)
    print("shape:",csv_data.shape)
    print("dtype:",csv_data.dtype)
