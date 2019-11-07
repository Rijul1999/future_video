# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:59:52 2019

@author: USER
"""

import numpy as np
import cv2
import skvideo.io

l = ["s08-d02-cam-002.avi", "s08-d04-cam-002.avi", "s08-d11-cam-002.avi", "s08-d14-cam-002.avi", "s10-d02-cam-002.avi"
     , "s10-d10-cam-002.avi", "s10-d11-cam-002.avi", "s11-d01-cam-002.avi", "s11-d06-cam-002.avi", "s11-d11-cam-002.avi"
     , "s11-d12-cam-002.avi", "s11-d13-cam-002.avi", "s11-d14-cam-002.avi", "s12-d05-cam-002.avi", "s12-d07-cam-002.avi"
     , "s12-d09-cam-002.avi", "s12-d10-cam-002.avi", "s12-d14-cam-002.avi", "s13-d08-cam-002.avi", "s13-d09-cam-002.avi"
     , "s13-d11-cam-002.avi", "s13-d12-cam-002.avi", "s13-d13-cam-002.avi", "s14-d08-cam-002.avi", "s14-d09-cam-002.avi"
     , "s14-d11-cam-002.avi", "s15-d03-cam-002.avi", "s15-d07-cam-002.avi", "s15-d14-cam-002.avi", "s16-d01-cam-002.avi"
     , "s16-d06-cam-002.avi", "s16-d09-cam-002.avi", "s16-d11-cam-002.avi", "s17-d02-cam-002.avi", "s17-d05-cam-002.avi"
     , "s17-d13-cam-002.avi", "s18-d11-cam-002.avi", "s19-d01-cam-002.avi", "s19-d06-cam-002.avi", "s19-d07-cam-002.avi"
     , "s19-d09-cam-002.avi", "s19-d10-cam-002.avi", "s19-d12-cam-002.avi", "s20-d07-cam-002.avi"]
flag=0   
for i in l:
    cap = cv2.VideoCapture(i)
    
    ret=True
    while(ret==True):
        ret,frame=cap.read()
        ret,frame2=cap.read()
        if(ret==False):
            break
        frame = cv2.resize(frame,(32,16))
        frame = np.expand_dims(frame,axis=0)
        frame2 = cv2.resize(frame2,(32,16))
        frame2 = np.expand_dims(frame2,axis=0)
        t = np.vstack((frame,frame2))
        
        
        
        count=2
        while(count%100 !=0 and ret==True):
            ret,frame=cap.read()
            if(ret==False):
                break
            frame = cv2.resize(frame,(32,16))
            frame = np.expand_dims(frame,axis=0)
            t = np.vstack((t,frame))
            count+=1
            
            """if(ret==True):
                cv2.imshow('Frame',frame)
                count+=1;
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                
            else:
                break
         """
        if(count%100!=0):
            break
        
        if(flag==0):
            train = np.expand_dims(t,axis=0)
            flag=1
        else:
            m = np.expand_dims(t,axis=0)
            train = np.vstack((train,m))
            print(train.shape)
            print(" In "+i)
    
    cap.release()
    cv2.destroyAllWindows()
