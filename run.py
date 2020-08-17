import cv2
import time
import numpy as np
import os
import json
from helper_funcs import *


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




def main():
    read_from_webcam(display_shape=display_shape)


if(__name__=="__main__") : main()