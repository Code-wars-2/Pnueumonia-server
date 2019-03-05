import numpy as np
import cv2
import tensorflow as tf
CATEGORIES=["NORMAL", "ABNORMAL"]

def prepare(filepath):
     IMG_SIZE =50
     img_array=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
     #img_array = tf.cast(img_array, tf.float32)
     new_array=cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
     return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def model(file):
         model=tf.keras.models.load_model('trainedxray.txt')

         prediction=model.predict([prepare(file)])

         print(prediction[0][0])

         return(CATEGORIES[int (prediction[0][0])])
