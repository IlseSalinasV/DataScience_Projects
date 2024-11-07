from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AvgPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
import numpy as np



def load_data():
    (features_train, target_train), (features_test, target_test) = fashion_mnist.load_data()

    features_train = features_train.reshape(features_train.shape[0], 28, 28, 1)
    features_test = features_test.reshape(features_test.shape[0], 28, 28, 1)

    features_train = features_train.astype('float32') / 255.0
    features_test =  features_test.astype('float32') / 255.0

    return features_train, target_train, features_test, target_test

def create_model():
    model = keras.models.Sequential()

    model.add(Conv2D(filters=6, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(AvgPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    optimizer = Adam()
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['acc'],
    )
    return model

def train_model(model, features_train, target_train):
    model.fit(
        features_train,
        target_train,
        epochs=10,
        verbose=2,
        validation_data=(features_test, target_test),
    )
    return model

features_train, target_train, features_test, target_test = load_data()
model = create_model()
model = train_model(model, features_train, target_train)