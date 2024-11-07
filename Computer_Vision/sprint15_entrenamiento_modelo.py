from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# original: load_data(path, subset)
def load_data(dataframe):
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1 / 255)

    train_datagen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345)

    val_datagen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345)

    return train_datagen_flow, val_datagen_flow

# original: create_model(input_shape)
def create_model():
    model = Sequential()
    backbone = ResNet50(
        input_shape=(224, 224, 3), weights='imagenet', include_top=False
    )

    model.add(backbone)

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Flatten())

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error']
    )
    return model

def train_model(model, train_data, test_data, batch_size, epochs, steps_per_epoch, validation_steps):
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=3,
        steps_per_epoch=len(train_datagen_flow),
        validation_steps=len(val_datagen_flow),
    )
    return model

labels = pd.read_csv('/datasets/faces/labels.csv')
train_datagen_flow, val_datagen_flow = load_data(labels)
model = create_model()

batch_size = 32
steps_per_epoch = train_datagen_flow.samples // batch_size
validation_steps = val_datagen_flow.samples // batch_size

# original: model = train_model(model, train_data, test_data, batch_size, epochs, steps_per_epoch, validation_steps)
# batch_size: comunmente se usan valores como 16, 32 o 64.
# steps_per_epoch: número total de imágenes dividido por el batch_size
model = train_model(model, train_datagen_flow, val_datagen_flow, batch_size, 3, steps_per_epoch, validation_steps)