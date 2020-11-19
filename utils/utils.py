import pandas as pd
from datetime import datetime
import os
import numpy as np

def create_meta_VLCS():
    def get_file_by_label(label_name):
        folder_root = "data/VLCS/Raw images/SUN/test/" + label_name
        rs = []
        paths = [] 
        labels = []
        for filename in os.listdir(folder_root):
            raw = []
            raw.append("SUN/test/"+label_name+"/" + filename)
            raw.append(label_name)
            rs.append(raw)
        
        return rs

    rs = []
    rs += get_file_by_label(label_name = "0")
    rs += get_file_by_label(label_name = "1")
    rs += get_file_by_label(label_name = "2")
    rs += get_file_by_label(label_name = "3")
    rs += get_file_by_label(label_name = "4")

    dataframe = pd.DataFrame(rs, columns = ['path', 'label'])
    dataframe.to_csv("SUN_test_kfold.txt",header=None, sep=" ", encoding='utf-8', index=False)

def get_results():
    results = [[76, 76, 96, 69, 79.25], [72, 74, 95, 65, 76.5], [71, 76, 96, 64, 76.75], [76, 75, 96, 68, 78.75], [76, 76, 96, 65, 78.25]]
    results = np.array(results)
    print(np.mean(results[:,4]))
    print(np.std(results[:,4]))
    
get_results()