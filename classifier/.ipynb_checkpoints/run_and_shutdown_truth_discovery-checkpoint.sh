#!/bin/bash

# 运行 Python 文件，并将输出同时写入 txt 文件和显示在终端
python truth_discovery_classifier_3_models.py 2>&1 | tee output_3models.txt
python truth_discovery_classifier_5_models.py 2>&1 | tee output_5models.txt
python truth_discovery_classifier_7_models.py 2>&1 | tee output_7models.txt

#关机
/usr/bin/shutdown now
#则关机
