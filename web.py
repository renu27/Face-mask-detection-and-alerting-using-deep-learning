# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:24:34 2021

@author: T M RENUSHREE
"""

import tensorflow as tf
import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import speech_recognition as sr
from playsound import playsound
import cv2
import os
import time
from gtts import gTTS
import matplotlib.pyplot as plt
#%matplotlib inline
#print(tf._version_)
model = tf.keras.models.load_model("Models/My_Model.h5")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #cap = cv2.VideoCapture("C:/Users/T M RENUSHREE/Desktop/face mask detection/examples/public_video2.mp4")
cap = cv2.VideoCapture(0)
    

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
writer = cv2.VideoWriter("output/sound_real_output1.mp4",cv2.VideoWriter_fourcc(*'DIVX'),25,(width,height))
    
while True:
    ret, img = cap.read()
    if ret == True:
        time.sleep(1/25)
    
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 8)
    
        for (x, y, w, h) in faces:
    
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask, withoutMask) = model.predict(face)[0]
            mask = mask*100
            withoutMask = withoutMask*100
    
            font = cv2.FONT_HERSHEY_SIMPLEX
    
                # Getting Text Size in pixel
                # print(\Image Width: \ , w)
            textSize = cv2.getTextSize(text="No Mask: " + str("%.2f" % round(mask, 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3)
                # print(\Text Width: \ , textSize[0][0])
            
            if mask > withoutMask:
                cv2.putText(img,
                                text = "Mask: " + str("%.2f" % round(mask, 2)),
                                org = (x-5,y-20),
                                fontFace=font,
                                fontScale = (2*w)/textSize[0][0],
                                color = (0, 255, 0),
                                thickness = 3,
                                lineType = cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 5)
            else:
                #data="please wear mask"
                #anguage="en"
                #myvar=gTTS(text=data,lang=language,slow=False)
                #myvar.save("C:/Users/T M RENUSHREE/Desktop/face mask detection/newfile10.mp3")
                #os.system("C:/Users/T M RENUSHREE/Desktop/face mask detection/beep.mp3")
                
                cv2.putText(img,
                                text = "No Mask: " + str("%.2f" % round(withoutMask, 2)),
                                org = (x-5,y-20),
                                fontFace=font,
                                fontScale = (1.8*w)/textSize[0][0],
                                color = (0, 0, 255),
                                thickness = 3,
                                lineType = cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)
                playsound('beep.mp3')
                    
                    
            
            # Storing Recorded File
        writer.write(img)
            
            # Display    
        cv2.imshow("Face Mask Detection",img)
    
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break
            
cap.release()
cv2.destroyAllWindows()