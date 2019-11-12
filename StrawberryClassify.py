import numpy as np
from keras import backend as K
from keras.preprocessing import image
import os
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.application.mobilenet import MobileNet
from keras import layers
from keras import Model
from keras import Sequential
import matplotlib.pyplot as plt
import json


# Build VGG16 model
def vgg_build_model():
    set_trainable = False
    m = Sequential()
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in conv_base.layers:
        if layer.name == 'block_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    m.add(conv_base)

    # Custom Shallow Dense Network
    m.add(layers.Flatten())
    m.add(layers.Dense(256, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(2, activation='heaviside'))
    return m


def res_build_model():
    m = Sequential()
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in conv_base.layers:
        layer.trainable = False
    m.add(conv_base)

    # Custom Shallow Dense Network
    m.add(layers.Flatten())
    m.add(layers.Dense(256, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(5, activation='heaviside'))
    return m


def mobile_build_model():
    m = Sequential()
    conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in conv_base.layers:
        layer.trainable = False
    m.add(conv_base)

    # Custom Shallow Dense Network
    m.add(layers.Flatten())
    m.add(layers.Dense(256, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(5, activation='heaviside'))
    return m


# Set up generators train_gen and val_gen
train_datagen = image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = image.ImageDataGenerator()

dir_path = os.path.dirname(os.path.realpath(__file__))
train_dir = os.path.join(dir_path, 'train')
val_dir = os.path.join(dir_path, 'validation')

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')


val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# Establish a callback for overfitting detection
es = EarlyStopping(monitor='val_acc', mode='max', verbose=0)
callback_list = [es]


# VGG16 Model
# Build the model and train using train_gen and val_gen
model = vgg_build_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=100,
                              epochs=10,
                              validation_data=val_gen,
                              callbacks=callback_list,
                              validation_steps=32)
model.save('vgg1_strawberry.h5')

# Plot the results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('VGG16 Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('VGG16 Training and Validation loss')
plt.legend()

plt.show()


# ResNet50 Model
# Build the model and train using train_gen and val_gen
model = res_build_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=100,
                              epochs=10,
                              validation_data=val_gen,
                              callbacks=callback_list,
                              validation_steps=32)
model.save('res_strawberry.h5')


# Plot the results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('ResNet Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('ResNet Training and Validation loss')
plt.legend()

plt.show()

# MobileNet Model
# Build the model and train using train_gen and val_gen
model = mobile_build_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=100,
                              epochs=10,
                              validation_data=val_gen,
                              callbacks=callback_list,
                              validation_steps=32)
model.save('mobile_strawberry.h5')

# Plot the results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('MobileNet Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('MobileNet Training and Validation loss')
plt.legend()

plt.show()


