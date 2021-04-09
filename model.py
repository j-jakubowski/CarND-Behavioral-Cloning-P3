
import cv2  
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import importInputData, plotHistogram
from random import shuffle


from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf

def generator(samples, batch_size=32):
    batch_size = int (batch_size / 2)
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample['image'])
                center_angle = batch_sample['steering']

                images.append(center_image)
                angles.append(center_angle)

                images.append(cv2.flip(center_image, 0))
                angles.append(-center_angle)


            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield tuple(sklearn.utils.shuffle(X_train, y_train))



def NvidiaBasedModel():

    def RGB2HSV(x):
        return tf.image.rgb_to_hsv(x)

    def RGB2YUV(x):
        return tf.image.rgb_to_yuv(x)

    activation = 'elu'
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name = 'Normalisation'))
    # model.add(Lambda(RGB2YUV))   

    # model.add(Dropout(0.3))
    model.add(Conv2D(24,kernel_size = (5,5), strides=(2,2),activation = activation))
    # model.add(Dropout(0.3))
    model.add(Conv2D(36,5,2,activation = activation))
    # model.add(Dropout(0.3))
    model.add(Conv2D(48,5,2,activation = activation))
    # model.add(Dropout(0.3))
    model.add(Conv2D(64,3,activation = activation))
    # model.add(Dropout(0.3))
    model.add(Conv2D(64,3,activation = activation))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100))  
    # model.add(Dropout(0.3))     
    model.add(Dense(50))
    # model.add(Dropout(0.3))
    model.add(Dense(10))
    # model.add(Dropout(0.3))
    model.add(Dense(1))
    return model



if __name__ == "__main__":
    sampleList = importInputData('./SampleData/', isCustom = False)

    dataSet =  sampleList 

    print('Number of unique images ' + str(len(dataSet)))

    plotHistogram(dataSet)


    train_samples, validation_samples = train_test_split(dataSet, test_size=0.35)

    batch_size=64

    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    #just flipping images 
    extraData = 1

    augmentedTrainSetLen = len(train_samples) * (1 + extraData)
    augementedValidSetLen = len(validation_samples) * (1 + extraData)

    print('Training set ' + str(augmentedTrainSetLen))
    print('Validation set ' + str(augementedValidSetLen))

    model = NvidiaBasedModel()
    model.compile(loss='mse', optimizer='adam')

    callback = EarlyStopping(monitor='val_loss', patience=3, min_delta = 0.001)

    history_object = model.fit(train_generator, 
                steps_per_epoch=np.ceil(augmentedTrainSetLen / batch_size), 
                validation_data=validation_generator, 
                validation_steps=np.ceil(augementedValidSetLen / batch_size), 
                epochs=7, callbacks=[callback], verbose=1)


    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save('model.h5')



