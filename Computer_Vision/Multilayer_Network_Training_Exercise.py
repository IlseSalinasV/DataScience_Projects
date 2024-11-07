from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    (features_train, target_train), (features_test, target_test) = fashion_mnist.load_data()

    features_train = features_train.reshape(features_train.shape[0], 28 * 28)
    features_test = features_test.reshape(features_test.shape[0], 28 * 28)

    features_train = features_train.astype('float32') / 255.0
    features_test =  features_test.astype('float32') / 255.0

    return features_train, target_train, features_test, target_test

def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=10, input_dim=28 * 28, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc'])
    return model

def train_model(model, features_train, target_train):
    model.fit(
        features_train,
        target_train,
        epochs=1,
        verbose=2,
        validation_data=(features_test, target_test),
    )
    return model

features_train, target_train, features_test, target_test = load_data()
model = create_model()
model = train_model(model, features_train, target_train)