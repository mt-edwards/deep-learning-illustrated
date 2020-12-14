# import dependencies
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, GRU, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

# model directory
model_dir = "models/gru"

# training
epochs = 4
batch_size = 128

# vector-space embedding
n_dim = 64
n_unique_words = 10000
max_review_length = 100
pad_type = trunc_type = 'pre'
drop_embed = 0.2

# neural network architecture
n_gru = 256
drop_gru = 0.2

# load data
(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words)

# padding and truncating data
x_train = pad_sequences(x_train,
                        maxlen=max_review_length,
                        padding=pad_type,
                        truncating=trunc_type,
                        value=0)
x_valid = pad_sequences(x_valid,
                        maxlen=max_review_length,
                        padding=pad_type,
                        truncating=trunc_type,
                        value=0)

# model specification
model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(SpatialDropout1D(drop_embed))
model.add(GRU(n_gru, dropout=drop_gru))
model.add(Dense(1, activation='sigmoid'))

# model summary
model.summary()

# mdoel compilation
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model checkpoint
modelcheckpoint = ModelCheckpoint(filepath=model_dir + "/weights.{epoch:02d}.hdf5")

# model fitting
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[modelcheckpoint])

# load model
model.load_weights(model_dir + "/weights.02.hdf5")

# model predictions
y_hat = model.predict(x_valid)

# model evaluation
roc_auc_score(y_valid, y_hat)
