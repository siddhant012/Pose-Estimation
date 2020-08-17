import cv2
import time
import numpy as np
import os
import json
import info
from helper_funcs import *


#record a single pose
def record_a_pose_encoding(display_shape):
    timer=5
    cam=cv2.VideoCapture(0)
    if not cam.isOpened() : raise IOError("Cannot open webcam")
    print("\nStand back.Recording your pose...")
    print("Press esc to exit.")
    for i in range(1,timer+1) : print("starting in ",timer+1-i,end="\r\r");time.sleep(1.0)
    print("\n")

    while True:
        _,frame=cam.read()
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
        cv2.imshow("recorded pose",frame)
        cv2.waitKey(1000)
        print("recorded pose.")

        encoding=get_encoding(points)
        break
    
    cam.release()
    cv2.destroyAllWindows()
    return encoding


#read images from webcam for pose estimation
def read_from_webcam(recorded_encoding,display_shape=(500,500)):

    timer=5
    cam=cv2.VideoCapture(0)
    if not cam.isOpened() : raise IOError("Cannot open webcam")
    print("\nRecording poses to compare with the recorded pose.")
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

        encoding=get_encoding(points)
        simmilarity=get_simmilarity(recorded_encoding,encoding)
        cv2.putText(frame,str(simmilarity),org=(display_shape[0]//2,display_shape[1]//2),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2,lineType=cv2.LINE_AA)

        frame=cv2.cvtColor(cv2.UMat(frame),cv2.COLOR_BGR2RGB)
        cv2.imshow("skeleton+bg",frame)
        cv2.waitKey(1)
        
    
    cam.release()
    cv2.destroyAllWindows()


def main():
    print("\nMake sure that you are in well contrast with the background.")
    encoding=record_a_pose_encoding(display_shape)
    read_from_webcam(encoding,display_shape)


if(__name__=="__main__") : main() 