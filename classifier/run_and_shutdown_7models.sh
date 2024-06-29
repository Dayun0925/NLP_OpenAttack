#!/bin/bash

# 运行 Python 文件，并将输出同时写入 txt 文件和显示在终端
python truth_discovery_classifier_7_models.py 2>&1 | tee output_7models.txt

# 如果 Python 文件执行成功，则关机
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    /usr/bin/shutdown now
else
    echo "Python script execution failed. Not shutting down."
fi
