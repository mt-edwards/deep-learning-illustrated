# %% Import Dependencies
import tensorflow
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# %% Load Data
(X_train, y_train), (X_valid, y_valid) = boston_housing.load_data()

# %% Model Specification
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(13,)))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# %% Model Summary
model.summary()

# %% Model Compilation
model.compile(loss='mean_squared_error',
              optimizer='adam')

# %% Model Fitting
model.fit(X_train, y_train,
          batch_size=8,
          epochs=32,
          verbose=1,
          validation_data=(X_valid, y_valid))

# %% Model Evaluation
model.evaluate(X_valid, y_valid)

# %%
