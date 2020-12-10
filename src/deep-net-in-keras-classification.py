# import dependencies
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard

# load data
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

# reformatting input data
X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')
X_train /= 255
X_valid /= 255

# reformatting output data
n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_valid = to_categorical(y_valid, n_classes)

# model specification
model = Sequential()

# first hidden dense layer with batch normalization
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(BatchNormalization())

# second hidden dense layer with batch normalization
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())

# third hidden dense layer with batch normalization and dropout
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# output dense layer
model.add(Dense(10, activation='softmax'))

# model summary
model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# tensor board configure
tensorboard = TensorBoard(log_dir='logs/deep-net')

# model fitting
model.fit(X_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(X_valid, y_valid),
          callbacks=[tensorboard])

# model evaluation
model.evaluate(X_valid, y_valid)
