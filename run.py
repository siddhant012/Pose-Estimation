import cv2
import time
import numpy as np
import os
import json
import info

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

#read images from webcam for pose estimation
def read_from_webcam(display_shape=(500,500)):

    timer=3
    cam=cv2.VideoCapture(0)
    if not cam.isOpened() : raise IOError("Cannot open webcam")
    print("\nEstimating Poses...")
    print("Press esc to exit.")
    for i in range(1,timer+1) : print("starting in ",timer+1-i,end="\r\r");time.sleep(1.0)
    print("\n")

    counter=0
    while True:
        _,frame=cam.read()
        counter+=1
        if(frame is None) : break
        frame=cv2.resize(frame,display_shape)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.uint8)

        points=get_points(frame)

        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
        
            if(points[partA] is None or points[partB] is None) : continue
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)

        frame=cv2.cvtColor(cv2.UMat(frame),cv2.COLOR_BGR2RGB)
        cv2.imshow("skeleton+bg",frame)
        cv2.waitKey(1)

        encoding=get_encoding(points)
        save_encoding(encoding,os.path.join(path,"webcam encodings/"+str(counter)+".json"))
    
    cam.release()
    cv2.destroyAllWindows()
    
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


def main():
    read_from_webcam(display_shape=display_shape)


if(__name__=="__main__") : main()