import numpy as np
import json
import info
import cv2
import time
import json
import os

mode=info.mode
path=info.path
display_shape=info.display_shape

if(mode=="mpi"):
    protoFile = os.path.join(path,"mpi/pose_deploy_linevec_faster_4_stages.prototxt")
    weightsFile = os.path.join(path,"mpi/pose_iter_160000.caffemodel")
    nPoints = 15
    threshold=0.1
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
elif(mode=="coco"):
    protoFile = os.path.join(path,"coco/pose_deploy_linevec.prototxt")
    weightsFile = os.path.join(path,"coco/pose_iter_440000.caffemodel")
    nPoints = 18
    threshold=0.1
    POSE_PAIRS=[ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
inWidth = 368
inHeight = 368


#save an image
def save_img(image,save_name):
    image=cv2.cvtColor(cv2.UMat(image),cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(save_name,image) : raise Exception("Could not write image")


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


#get the encodings , given the skeletal joints of a person in an image
def get_encoding(points):

    def sigmoid(x):
        if(x<-80):return 0
        elif(x>80):return 1
        return 1/(1+np.exp(-x))

    eps=1.0e-8
    
    if(points[0] is None):
        encoding=[0]*len(points)
    else:
        headx,heady=points[0]
        temp=[(point[0]-headx,point[1]-heady) if point is not None else None for point in points]
        encoding=[0]+[sigmoid(point[1]/(point[0]+eps)) if point is not None else 0 for point in temp[1:]]
    return encoding


#get the skeletal joints from an image given in appropriate format
def get_points(image):
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
        
    H = output.shape[2]
    W = output.shape[3]

    points = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (image.shape[1] * point[0]) / W
        y = (image.shape[0] * point[1]) / H
 
        if(prob>threshold) : points.append((int(x), int(y)))
        else : points.append(None)

    return points


#save the encodings in json format
def save_encoding(encoding,save_name):
    encoding=list(encoding)
    encoding_json={'encoding':encoding}
    encoding_json=json.dumps(encoding_json)
    with open(save_name,"w") as file : file.write(encoding_json)