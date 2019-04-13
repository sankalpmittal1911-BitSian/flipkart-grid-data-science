import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import glob
import cv2
import imageio
import os.path as path
from scipy import misc
import csv
from IPython.display import Image
from google.colab import drive
drive.mount('/content/drive')
import tensorflow as tf
#tf.enable_eager_execution()
import random
 
 
 
IMAGE_PATH = '/content/drive/My Drive/flipkart data science contest/training-images'
with open('/content/drive/My Drive/flipkart data science contest/training_data.csv') as csvfile:
    data = list(csv.reader(csvfile))
 
n_images=len(data)-1
# Load the images and Read the labels from the filenames
y_samples = []
x=[]
X_samples=[]
for i in range(1,14001):
  print(i)
  X_samples.append(path.join(IMAGE_PATH,str(data[i][0])))
  x.append(float(data[i][5]))
  x.append(float(data[i][6]))
  x.append(float(data[i][7]))
  x.append(float(data[i][8]))
  x.append(float(data[i][9]))
  y_samples.append(x)
  x=[]
  
 
c = list(zip(X_samples, y_samples))
 
random.shuffle(c)
 
X_samples, y_samples = zip(*c)
 
X_valid_samples=X_samples[0:3500]
y_valid_samples=y_samples[0:3500]
 
X_train_samples=X_samples[3500:]
y_train_samples=y_samples[3500:]
 
 
 
def DATA_GEN(X_samples, y_samples, batch_size):
  while True:
    #batch_size = len(X_samples) // batch_size
    X_batches = [X_samples[i:min(i+batch_size,len(X_samples))] for i in range(0, len(X_samples), batch_size)] #np.split(X_samples, batch_size)
    y_batches = [y_samples[j:min(j+batch_size,len(y_samples))] for j in range(0, len(y_samples), batch_size)] #np.split(y_samples, batch_size)
    for b in range(len(X_batches)):
      z=[]
      #if(b==len(X_batches)-1):
        #val=len(X_batches) - 16*(len(X_batches)//16)
      #else:
    #batch_size = len(X_samples) // batch_size
      for c in range(batch_size):
        x = imageio.imread(X_batches[b][c])
        z.append(x)
      x=np.array(z)
      x=x/255
      y = np.array(y_batches[b])
      #print(x.shape)
      #print(y.shape)
      yield x, y
     
    
 
 
datagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=True, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=0, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0, # randomly shift images vertically (fraction of total height)
        horizontal_flip=False, # randomly flip images
        vertical_flip=False) # randomly flip images
 
 
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
 
 
#sess = tf.Session()
 
#strategy = tf.distribute.MirroredStrategy()
 
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
#Let's create our model for training
 
def space_to_depth_x2(x):
    
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)
 
 
def space_to_depth_x2_output_shape(input_shape):
   
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])
  
  
def cnn():
  
  #model = Sequential()
  
  inp=Input(shape=(480,640,3))
  
  x=Conv2D(32, kernel_size=(1, 1))(inp)
  #x=Reshape((480, 640, 32))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
  x=MaxPooling2D(pool_size=(2, 2))(x)
 
  x=(Conv2D(64, kernel_size=(1, 1)))(x)
  x=(BatchNormalization())(x)
  x=LeakyReLU()(x)
  x=(MaxPooling2D(pool_size=(2, 2)))(x)
 
 
  x=Conv2D(128, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
 
  x=Conv2D(64, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
 
  x=Conv2D(128, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
  x=MaxPooling2D(pool_size=(2, 2))(x)
 
  x=Conv2D(256, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
 
  x=Conv2D(128, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(256, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
  x=MaxPooling2D(pool_size=(2, 2))(x)
 
  x=Conv2D(512, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(256, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)         
 
  x=Conv2D(512, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(256, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(512, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  y=LeakyReLU()(x)
  x=MaxPooling2D(pool_size=(2, 2))(y)
 
  x=Conv2D(1024, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(512, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(1024, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(512, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(1024, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(1024, kernel_size=(1, 1))(x)
  a=BatchNormalization()(x)
            
  b=Conv2D(64, kernel_size=(1, 1))(y)
  
  z=LeakyReLU()(a)
  
  x=BatchNormalization()(b)
 
  p=Conv2D(1024, kernel_size=(1, 1))(z)
  
  q=LeakyReLU()(x)
  
  r=BatchNormalization()(p)
  
  conv21_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(q)
  
  
  s=LeakyReLU()(r)
  
  x = concatenate([conv21_reshaped, s])
  
  x=Conv2D(1024, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(425, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(25, kernel_size=(1, 1))(x)
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
 
  x=Conv2D(5, kernel_size=(1, 1))(x)
  
  x=BatchNormalization()(x)
  x=LeakyReLU()(x)
  
  x=Conv2D(5, kernel_size=(1, 6))(x)
  
  x=Flatten()(x)
  
  x=Dense(5)(x)
  
  x=BatchNormalization()(x)
  
  preds = LeakyReLU()(x)
 
  model=Model(inp,preds)
  
  model.compile(optimizer='adam', loss='mse', metrics=[mAPmetric])
#optimizer=tf.train.AdamOptimizer(0.001)
            
  model.summary()
            
  return model
k=0
model=cnn()
#print(len(X_samples))
#print(len(y_samples))
n_epoch = 100
 
#for e in range(n_epoch):
 # print ("epoch : ", e)
  #for X_train, y_train in DATA_GEN(X_samples, y_samples): # chunks of 100 images
    #print(DATA_GEN(X_samples, y_samples))
    #print(X_train.shape)
    #print(y_train.shape)
    #print(X_train)
    #print(y_train)
    #for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=16): # chunks of 32 samples
      #print(datagen.flow(X_train, y_train, batch_size=16))
      #loss = model.train_on_batch(X_batch, y_batch)
     # k=k+1
      #print(k)
     # model.fit(X_train,y_train,verbose=2, validation_split=0.25, shuffle= True)
      
      #print(loss)
      
import keras.callbacks
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
 
callbacks = [
    EarlyStopping(monitor='val_loss',patience=10, verbose=1),
    ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, min_lr=0.00000001, verbose=1),
    ModelCheckpoint('/content/drive/My Drive/model_yolo.h5', monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False)
]
 
 
model.fit_generator(DATA_GEN(X_train_samples, y_train_samples, batch_size=12), steps_per_epoch=875 , epochs=100, verbose=1 , callbacks=callbacks, validation_data = DATA_GEN(X_valid_samples,y_valid_samples,batch_size=4) , validation_steps = 875 )

#Evaluate on training and validation set

model.evaluate_generator(DATA_GEN(X_train_samples, y_train_samples, batch_size=12))
model.evaluate_generator(DATA_GEN(X_valid_samples,y_valid_samples,batch_size=4))

#Now you can test on any image
#model.predict_generator(test_generator)

model.save('/content/drive/My Drive/flipkart data science contest/model_yolo.h5')
