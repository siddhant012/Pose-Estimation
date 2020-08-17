import cv2
import time
import numpy as np
import os
import json
import info
from helper_funcs import *


def start():
    dir=os.path.join(path,"samples/")
    frames_num=700
    for i in range(1,frames_num+1) : os.mkdir(dir+"/"+str(i))
    cap=cv2.VideoCapture(os.path.join(dir,"far.mp4"))
    if not cap.isOpened() : raise IOError("Cannot open file")
    
    count=0
    while True:
        print("done=",count," frames",end="\r\r");count+=1
        _,frame=cap.read()
        if(frame is None) : break
        frame=cv2.resize(frame,display_shape)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.uint8)

        points=get_points(frame)

        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
        
            if(points[partA] is None or points[partB] is None) : continue
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)

        encoding=get_encoding(points)
        save_img(frame,os.path.join(dir,str(count)+"/"+"far_frame.jpg"))
        save_encoding(encoding,os.path.join(dir,str(count)+"/"+"far_encoding.json"))
        if(count>=frames_num) : break
    

    cap=cv2.VideoCapture(os.path.join(dir,"near.mp4"))
    if not cap.isOpened() : raise IOError("Cannot open file")
    
    count=0
    while True:
        print("done=",count," frames",end="\r\r");count+=1
        _,frame=cap.read()
        if(frame is None) : break
        frame=cv2.resize(frame,display_shape)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.uint8)

        points=get_points(frame)

        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
        
            if(points[partA] is None or points[partB] is None) : continue
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)

        encoding=get_encoding(points)
        save_img(frame,os.path.join(dir,str(count)+"/"+"near_frame.jpg"))
        save_encoding(encoding,os.path.join(dir,str(count)+"/"+"near_encoding.json"))
        if(count>=frames_num):break

    
    for count in range(1,frames_num+1):
        far_encoding=load_encoding(os.path.join(dir,str(count+1)+"/"+"far_encoding.json"))
        near_encoding=load_encoding(os.path.join(dir,str(count+1)+"/"+"near_encoding.json"))

        simmilarity=get_simmilarity(far_encoding,near_encoding)
        with open(os.path.join(dir,str(count)+"/"+"simmilarity.txt"),"w") as file : file.write(str(simmilarity))
    
    cap.release()
    cv2.destroyAllWindows()
    print("completed")



def main():
    start()

if(__name__=="__main__") : main()
