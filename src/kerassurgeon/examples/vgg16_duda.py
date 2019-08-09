import keras
import tensorflow as tf
import os
import sklearn.metrics as metrics
import numpy as np
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras import callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, \
    LearningRateScheduler, TensorBoard
from kerassurgeon.operations import delete_layer, insert_layer, \
    delete_channels, replace_layer
from kerassurgeon import Surgeon
from operator import itemgetter, attrgetter
from keras.utils import np_utils

from kerassurgeon import identify

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

# training parameters
batch_size = 128
maxepoches = 300
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

training_verbosity = 2

# VGG model
model = Sequential()
weight_decay = 0.0005

model.add(Conv2D(64, (3, 3), padding='same', input_shape=[32, 32, 3], name= 'conv1-1',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same', name= 'conv1-2',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', name= 'conv2-1',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same', name= 'conv2-2',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(256, (3, 3), padding='same', name= 'conv3-1',
#                  kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Conv2D(256, (3, 3), padding='same', name= 'conv3-2',
#                  kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Conv2D(256, (3, 3), padding='same', name= 'conv3-3',
#                  kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(512, (3, 3), padding='same', name= 'conv4-1',
#                  kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Conv2D(512, (3, 3), padding='same', name= 'conv4-2',
#                  kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Conv2D(512, (3, 3), padding='same', name= 'conv4-3',
#                  kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', name= 'conv5-1',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', name= 'conv5-2',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', name= 'conv5-3',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('softmax'))
print('\033[93m'+"Model created"+'\033[0m')

# optimization details
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9,
                     nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])
print('\033[93m'+"Model compile"+'\033[0m')
#model.summary()

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_test_arr = y_test
x_train /= 255.
x_test /= 255.
y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)

#train cifar100 on VGG16
early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=10,
                                            verbose=training_verbosity,
                                            mode='auto')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,
                                        patience=5,
                                        verbose=training_verbosity,
                                        mode='auto',
                                        epsilon=0.0001,
                                        cooldown=0,
                                        min_lr=0)

results = model.fit(x_train,
                        y_train,
                        epochs=1,
                        batch_size=128,
                        verbose=2,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, reduce_lr])


print('TRAINING IS FINISHED!!!')
loss = model.evaluate(x_test, y_test, batch_size=128, verbose=2)
print('\033[93m'+'original model loss:' + str(loss) + '\033[0m'+  '\n')

for layer in model.layers:
    apoz = identify.get_apoz(model, layer, x_test)
    high_apoz_channels = identify.high_apoz(apoz)
    model = delete_channels(model, layer, high_apoz_channels)

    print('layer name: ', layer.name)

    model.compile(optimizer=sgd,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    loss = model.evaluate(x_test,
                            y_train,
                            batch_size=128,
                            verbose=2)
    print('model loss after pruning: ', loss, '\n')

    results = model.fit(mnist.train.images,
                        mnist.train.labels,
                        epochs=10,
                        batch_size=128,
                        verbose=training_verbosity,
                        validation_data=(x_test, y_train),
                        callbacks=[early_stopping, reduce_lr])

    loss = model.evaluate(x_test,
                            y_train,
                            batch_size=128,
                            verbose=2)
    print('\033[93m','model loss after retraining: ', loss, '\033[0m','\n')
    




# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,
    # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,
    # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)


# learning rate
def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


reduce_lr = LearningRateScheduler(lr_scheduler)

# checkpoint
model_checkpoint = ModelCheckpoint("cifar100vgg.h5", monitor="val_acc",
                                   save_best_only=True,
                                   save_weights_only=False, verbose=1)

# Tensorboard
tbCallBack = TensorBoard(log_dir='./Graph-C100-VGG', histogram_freq=0,
                         write_graph=True, write_images=True)

# training process in a for loop with learning rate drop every 25 epoches.
historytemp = model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=maxepoches,
    validation_data=(x_test, y_test),
    callbacks=[reduce_lr, model_checkpoint, tbCallBack], verbose=1)

# acc after training
yPreds = model.predict(x_test, verbose=1)
yPred = np.argmax(yPreds, axis=1)
yTrue = y_test_arr

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy: ", accuracy)
print("Error: ", error)
