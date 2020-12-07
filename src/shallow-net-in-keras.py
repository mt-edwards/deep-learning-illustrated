# %% Import Dependencies
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt

# %% Load Data
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

# %% Reformatting Input Data
X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')
X_train /= 255
X_valid /= 255

# %% Reformatting Output Data
n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_valid = to_categorical(y_valid, n_classes)

# %% Model Spesification
model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax',))

# %% Model Summary
model.summary()

# %% Model Compilation
model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

# %% Model Fitting
model.fit(X_train, y_train,
          batch_size=128,
          epochs=200,
          verbose=1,
          validation_data=(X_valid, y_valid))

# %% Model Evaluation
model.evaluate(X_valid, y_valid)

# %%
