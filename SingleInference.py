import os
import sys
import cv2
import tensorflow as tf
import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
print("\n===\nLoading Model\n===\n")
tick=time.time()
model=tf.keras.models.load_model('Trained.h5')
print(f"\n===\nModel Loaded in {round((time.time()-tick),2)} seconds \n===\n")
argumentList = sys.argv
if len(argumentList)!=2:
    print(f"\n\n\n\nERROR\nExpected 1 arguement other than file name but received {len(argumentList)-1}\nPLEASE RERUN\n\n")
    exit()
else:
    tick=time.time()
    img=cv2.imread(argumentList[-1])
    img=np.resize(img,(1,256,256,3))
    img=(1./255.)*img

    pred=model.predict(img)
    predword=["Diseased" if pred<0.5 else "Healthy"][0]
    

    print(f"\nPrediction of model is:\n{predword}, with a sureity of {pred[0]}\nTime Taken:{round((time.time()-tick),2)} seconds")
