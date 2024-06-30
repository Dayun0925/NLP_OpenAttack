import sys
import numpy as np
from type_utils import *


def make_noise(category_num,size, otype=OUTPUT_TYPE.TYPE_LABEL):
    ran = np.random.rand(size,category_num)
    preds = softmax_2d(ran)
    if otype == OUTPUT_TYPE.TYPE_LABEL:
        preds = convert_to_type_label(preds)
    elif otype == OUTPUT_TYPE.TYPE_RANK:
        preds = convert_to_type_rank(preds)
    return preds
    

def produce(category_num, q_size, task, model, output_type):
    save_path = "outputs/%s/query_%d/type_%s/%s_noise.csv"%(task,q_size,output_type, model)
    otype = None
    if output_type == "label":
        otype = OUTPUT_TYPE.TYPE_LABEL
    elif output_type == "rank":
        otype = OUTPUT_TYPE.TYPE_RANK
    elif output_type == "probs":
        otype = OUTPUT_TYPE.TYPE_PROBS
    assert otype is not None
    preds = make_noise(category_num, q_size, otype)
    print("saving to %s..."%(save_path))
    save_results_to_txt(preds,save_path)

if __name__ == "__main__":

    t20News_models = {
        "knn",            
        "svm",            
        "decision_tree",  
        "srandom_forest", 
        "bagging",        
        "boost",          
        "DNN",            
        "CNN",            
        "RNN" ,           
        "RCNN"    
    }
    t20News_qsizes = [1000,3000,5000]
    t20News_category_num = 20

    output_types = ["probs"]
            produce(MNIST_category_num, q_size, "MNIST", model , output_type)

    # 20News
    for output_type in output_types:
        for q_size in t20News_qsizes:
            for model in t20News_models:
                produce(t20News_category_num, q_size, "20News", model , output_type)




        
        
        
        
        
        
        
        
        
        
















