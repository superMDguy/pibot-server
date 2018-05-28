from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Convolution2D, Input, Dropout, Flatten
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd

which_train = 3000
X = np.load('./training_data/frames_{}.npy'.format(which_train))
controls = pd.read_csv('./training_data/steer_{}.csv'.format(which_train))
y = [to_categorical(np.array(controls.drive).astype(np.uint8), num_classes=3),
     to_categorical(np.array(controls.steer).astype(np.uint8), num_classes=3)]


def norm_dropout(model):
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)
    return model


inputs = Input(shape=X[0].shape)
batchnorm = BatchNormalization()(inputs)
conv_1 = Convolution2D(24, (5, 5), activation='elu',
                       name='conv_1', strides=(2, 2))(batchnorm)
conv_1 = norm_dropout(conv_1)
conv_1 = BatchNormalization()(conv_1)
conv_2 = Convolution2D(36, (5, 5), activation='elu',
                       name='conv_2', strides=(2, 2))(conv_1)
conv_2 = norm_dropout(conv_2)
conv_3 = Convolution2D(48, (5, 5), activation='elu',
                       name='conv_3', strides=(2, 2))(conv_2)
conv_3 = norm_dropout(conv_3)
conv_4 = Convolution2D(64, (3, 3), activation='elu',
                       name='conv_4', strides=(1, 1))(conv_3)
conv_4 = norm_dropout(conv_4)
conv_5 = Convolution2D(64, (3, 3), activation='elu',
                       name='conv_5', strides=(1, 1))(conv_4)
conv_5 = norm_dropout(conv_5)

flat = Flatten()(conv_5)

dense_1 = Dense(1164)(flat)
dense_1 = norm_dropout(dense_1)
dense_2 = Dense(100, activation='elu')(dense_1)
dense_2 = norm_dropout(dense_2)
dense_3 = Dense(50, activation='elu')(dense_2)
dense_3 = norm_dropout(dense_3)
dense_4 = Dense(10, activation='elu')(dense_3)
dense_4 = norm_dropout(dense_4)

pred_drive = Dense(3, activation='softmax', name='pred_drive')(dense_4)
pred_steer = Dense(3, activation='softmax', name='pred_steer')(dense_4)

full_model = Model(inputs=[inputs], outputs=[pred_drive, pred_steer])

full_model.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
full_model.fit(X, y, epochs=100, validation_split=0.1,
               callbacks=[EarlyStopping(patience=2), ReduceLROnPlateau()])
full_model.fit(X, y, epochs=5)  # fine tune on all data
print('Saving model')
full_model.save('./trained.h5')
