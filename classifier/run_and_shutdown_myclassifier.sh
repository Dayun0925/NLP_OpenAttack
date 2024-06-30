#!/bin/bash

# 运行 Python 文件，并将输出同时写入 txt 文件和显示在终端
python myClassifier.py -model_type cnn -attacker_type DeepWordBug 2>&1 | tee myClassfier_output_cnn.txt
python myClassifier.py -model_type boost -attacker_type DeepWordBug 2>&1 | tee myClassfier_output_boost.txt
python myClassifier.py -model_type decision_tree -attacker_type DeepWordBug 2>&1 | tee myClassfier_output_decision_tree.txt
python myClassifier.py -model_type knn -attacker_type DeepWordBug 2>&1 | tee myClassfier_output_knn.txt
python myClassifier.py -model_type rcnn -attacker_type DeepWordBug 2>&1 | tee myClassfier_output_rcnn.txt
python myClassifier.py -model_type svm -attacker_type DeepWordBug 2>&1 | tee myClassfier_output_svm.txt
python myClassifier.py -model_type random_forest -attacker_type DeepWordBug 2>&1 | tee myClassfier_output_random_forest.txt
python myClassifier.py -model_type bagging -attacker_type DeepWordBug 2>&1 | tee myClassfier_output_bagging.txt
python myClassifier.py -model_type dnn -attacker_type DeepWordBug 2>&1 | tee myClassfier_output_dnn.txt

# 如果 Python 文件执行成功，则关机
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    /usr/bin/shutdown now
else
    echo "Python script execution failed. Not shutting down."
fi
