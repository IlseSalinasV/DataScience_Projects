from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(path):
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1 / 255)

    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345)

    val_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='validation',
        seed=12345)

    return train_datagen_flow, val_datagen_flow

def create_model():
    model = Sequential()
    backbone = ResNet50(
        input_shape=(150, 150, 3), weights='imagenet', include_top=False
    )

    model.add(backbone)

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Flatten())

    optimizer = Adam()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def train_model(model, train_datagen_flow, val_datagen_flow):
    model.fit(
        train_datagen_flow,
        validation_data=val_datagen_flow,
        epochs=3,
        steps_per_epoch=len(train_datagen_flow),
        validation_steps=len(val_datagen_flow),
    )
    return model

path = '/datasets/fruits_small/'
train_datagen_flow, val_datagen_flow = load_data(path)
model = create_model()
model = train_model(model, train_datagen_flow, val_datagen_flow)