# import dependencies
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Conv1D
from tensorflow.keras.layers import GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
# model directory
model_dir = "models/multi-conv"

# training
epochs = 4
batch_size = 128

# vector-space embedding
n_dim = 64
n_unique_words = 5000
max_review_length = 400
pad_type = trunc_type = 'pre'
drop_embed = 0.2

# convolutional layer architecture
n_conv = 256
k_conv_1 = 2
k_conv_2 = 3
k_conv_3 = 4

# dense layer architecture
n_dense_1 = 256
n_dense_2 = 64
dropout = 0.2

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
input = Input(shape=(max_review_length, ), dtype='int16', name='input')
embedding = Embedding(n_unique_words, n_dim, name='embedding')(input)
drop_embedding = SpatialDropout1D(drop_embed, name='drop_embedding')(embedding)
conv_1 = Conv1D(n_conv, k_conv_1, activation='relu',name='conv_1')(drop_embedding)
maxp_1 = GlobalMaxPooling1D(name='maxp_1')(conv_1)
conv_2 = Conv1D(n_conv, k_conv_2, activation='relu', name='conv_2')(drop_embedding)
maxp_2 = GlobalMaxPooling1D(name='maxp_2')(conv_2)
conv_3 = Conv1D(n_conv, k_conv_3, activation='relu', name='conv_3')(drop_embedding)
maxp_3 = GlobalMaxPooling1D(name='maxp_3')(conv_3)
concat = concatenate([maxp_1, maxp_2, maxp_3])
dense_1 = Dense(n_dense_1, activation='relu', name='dense_1')(concat)
drop_dense_1 = Dropout(dropout, name='drop_dense_1')(dense_1)
dense_2 = Dense(n_dense_2, activation='relu', name='dense_2')(drop_dense_1)
drop_dense_2 = Dropout(dropout, name='drop_dense_2')(dense_2)
predictions = Dense(1, activation='sigmoid', name='output')(drop_dense_2)
model = Model(input, predictions)

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
model.load_weights(model_dir + "/weights.03.hdf5")

# model predictions
y_hat = model.predict(x_valid)

# model evaluation
roc_auc_score(y_valid, y_hat)
