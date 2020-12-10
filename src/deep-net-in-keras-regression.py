# import dependencies
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# load data
(X_train, y_train), (X_valid, y_valid) = boston_housing.load_data()

# model specification
model = Sequential()

# first hidden dense layer with batch normalization
model.add(Dense(32, activation='relu', input_shape=(13,)))
model.add(BatchNormalization())

# second hidden dense layer with batch normalization and dropout
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# output dense layer
model.add(Dense(1, activation='linear'))

# model summary
model.summary()

# model compilation
model.compile(loss='mean_squared_error',
              optimizer='adam')

# model fitting
model.fit(X_train, y_train,
          batch_size=8,
          epochs=32,
          verbose=1,
          validation_data=(X_valid, y_valid))

# model evaluation
model.evaluate(X_valid, y_valid)
