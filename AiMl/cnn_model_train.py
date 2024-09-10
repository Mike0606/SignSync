import numpy as np
import pickle
import cv2
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from keras import backend as K
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    img = cv2.imread('gestures/1/100.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len([d for d in os.listdir('gestures') if os.path.isdir(os.path.join('gestures', d))])

image_x, image_y = get_image_size()

def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = optimizers.SGD(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "cnn_model_keras2.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    return model, callbacks_list

class TrainingProgressLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}:")
        print(f" - Loss: {logs['loss']:.4f}")
        print(f" - Accuracy: {logs['accuracy']:.4f}")
        print(f" - Validation Loss: {logs['val_loss']:.4f}")
        print(f" - Validation Accuracy: {logs['val_accuracy']:.4f}")

def train():
    print("Loading training data...")
    with open("train_images.pkl", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels.pkl", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    print("Loading validation data...")
    with open("val_images.pkl", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels.pkl", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    print("Loading test data...")
    with open("test_images.pkl", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels.pkl", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    num_classes = get_num_of_classes()

    print("Unique train labels:", np.unique(train_labels))
    print("Unique validation labels:", np.unique(val_labels))
    print("Unique test labels:", np.unique(test_labels))

    print("Reshaping data...")
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
    train_labels = np_utils.to_categorical(train_labels, num_classes=num_classes)
    val_labels = np_utils.to_categorical(val_labels, num_classes=num_classes)
    test_labels = np_utils.to_categorical(test_labels, num_classes=num_classes)

    print("Training model...")
    model, callbacks_list = cnn_model()
    model.summary()
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=100, batch_size=32, callbacks=callbacks_list)
    
    print("Evaluating model...")
    val_scores = model.evaluate(val_images, val_labels, verbose=0)
    print("Validation set accuracy: %.2f%%" % (val_scores[1]*100))

    print("Evaluating on test data...")
    test_scores = model.evaluate(test_images, test_labels, verbose=0)
    print("Test set accuracy: %.2f%%" % (test_scores[1]*100))

train()
K.clear_session()
