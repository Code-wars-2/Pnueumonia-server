import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
Name= 'xray-cnn'
pickle_in=open("X.txt","rb")
X=pickle.load(pickle_in)

pickle_in=open('Y.txt','rb')
Y=pickle.load(pickle_in)

X=X/255.0

model=Sequential()
'''
model.add(Conv2D(256, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3) , input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
'''
model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=X.shape[1:]))
model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=X.shape[1:]))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
'''
model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
'''
model.add(Flatten())

model.add(Dense(64))
model.add(Dropout(0.4))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard=TensorBoard(log_dir="logs/{}".format(Name))

model.compile( optimizer ='adam', metrics=['accuracy'], loss='binary_crossentropy')

model.fit(X,Y,batch_size=32, epochs=6, validation_split=0.3, callbacks=[tensorboard])
model.save('trainedxray.txt')
