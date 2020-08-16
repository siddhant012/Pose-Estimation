import numpy as np
import json

#load a pose encoding saved in json format
def load_encoding(save_name):
    with open(save_name,"r") as file : encoding=json.load(file)
    encoding=encoding['encoding']
    return np.array(encoding)

#get the simmilarity between two pose encodings
def get_simmilarity(encoding1,encoding2):
    encoding1=np.array(encoding1)
    encoding2=np.array(encoding2)
    
    c1,c2=0,0
    for i in encoding1: 
        if(i>0.01 and i<0.99):c1+=1
    for i in encoding2:
        if(i>0.01 and i<0.99):c2+=1

    return 1-(abs(encoding1-encoding2)).sum()/max(c1,c2)