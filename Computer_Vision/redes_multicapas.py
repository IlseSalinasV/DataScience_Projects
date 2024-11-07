from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
#import numpy as np


# load data
def load_data(datos):
    (features_train, target_train), (features_test, target_test) = datos.load_data()

    return features_train, target_train, features_test, target_test


X_train, y_train, X_test, y_test = load_data(fashion_mnist)

print(X_train.shape)
print(X_test.shape)

# reshape
X_train = X_train.reshape(X_train.shape[0], 28 * 28) / 255.0
X_test = X_test.reshape(X_test.shape[0], 28 * 28) / 255.0

print(X_train.shape)
print(X_test.shape)


# funcion creacion de modelo
def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=20,
                                 input_dim=28 * 28,
                                 activation='softmax'
                                 )
              )
    model.add(keras.layers.Dense(units=15,
                                 input_dim=28 * 28,
                                 activation='softmax'
                                 )
              )
    model.add(keras.layers.Dense(units=10,
                                 input_dim=28 * 28,
                                 activation='softmax'
                                 )
              )

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['acc']
                  )
    return model


# funcion entrenar modelo
def train_model(modelo, features_train, target_train, features_test, target_test):
    model.fit(features_train,
              target_train,
              epochs=1,
              verbose=2,
              validation_data=(features_test, target_test)
              )


model = create_model()
train_model = train_model(model, X_train, y_train, X_test, y_test)