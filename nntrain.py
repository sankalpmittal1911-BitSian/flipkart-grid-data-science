from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D,merge, Input,LeakyReLU,Reshape
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.models import Model
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
'''
from IPython.display import Image
from google.colab import drive
drive.mount('/content/drive')'''
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
 
 
def castF(x):
    return K.cast(x, K.floatx())
 
def castB(x):
    return K.cast(x, bool)
 
def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)
 
    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())
 
def mAPmetric(true, pred): #any shape can go - can't be a loss function
 
    tresholds = [0.5 + (i*.05)  for i in range(10)]
 
    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))
 
    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)
 
    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))    
    pred1 = castF(K.greater(predSum, 1))
 
    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)
 
    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)
 
    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred) 
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]
 
    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)
 
    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 
 
    return (truePositives + trueNegatives) / castF(K.shape(true)[0])
 
from keras.models import load_model
 
model=load_model('/home/blade/flipkart-work/model_yolo.h5')
 
#model.compile(optimizer='adam', loss='mse', metrics=[mAPmetric])
 
 
import csv
from os import path
import imageio
import numpy as np
 
def DATA_GEN2(X_samples, batch_size=1):
  #batch_size = len(X_samples) // batch_size
  X_batches = [X_samples[i:min(i+batch_size,len(X_samples))] for i in range(0, len(X_samples), batch_size)] #np.split(X_samples, batch_size)
  for b in range(len(X_batches)):
    z=[]
    #if(b==len(X_batches)-1):
      #val=len(X_batches) - 16*(len(X_batches)//16)
    #else:
      #val=16
    for c in range(1):
      x = imageio.imread(X_batches[b][c])
      z.append(x)
    x=np.array(z)
    x=x/255
    #print(X_batches)
    yield x
    
path1='/home/blade/flipkart-work/test-images'    
with open('/home/blade/flipkart-work/test.csv') as csvfile:
    data2 = list(csv.reader(csvfile))    
 
X_test_sample=[]
 
for i in range(1,12816):
  #print(i)
  X_test_sample.append(path.join(path1,str(data2[i][0])))
  #print(X_test_sample)
  
 
  # Gives prediction
 
with open('result.txt','a') as f :
  for X_test in DATA_GEN2(X_test_sample):
    print(model.predict(X_test))
    f.write("%s\n" % str(model.predict(X_test)[len(model.predict(X_test)-4,len(model.predict(X_test)))]))
