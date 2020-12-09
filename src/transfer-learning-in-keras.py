# import dependencies
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# two iamge generator classes
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   data_format="channels_last",
                                   rotation_range=30,
                                   horizontal_flip=True,
                                   fill_mode='reflect')
valid_datagen = ImageDataGenerator(rescale=1.0/255,
                                   data_format='channels_last')

# batch size
batch_size = 32

# define two data generators
train_generator = train_datagen.flow_from_directory(
    directory='./data/hot-dog-not-hot-dog/train',
    target_size=(224, 224),
    classes=['hot_dog', 'not_hot_dog'],
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=42)
valid_generator = valid_datagen.flow_from_directory(
    directory='./data/hot-dog-not-hot-dog/test',
    target_size=(224, 224),
    classes=['hot_dog', 'not_hot_dog'],
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=42)

# load pre-trained VGG19 model
vgg19 = VGG19(include_top=False,
              weights='imagenet',
              input_shape=(224, 224, 3),
              pooling=None)

# freeze all layers in pre-trained VGG19 models
for layer in vgg19.layers:
    layer.trainable = False

# model specification
model = Sequential()
model.add(vgg19)
model.add(Flatten(name='flatten'))
model.add(Dropout(0.5, name="dropout"))
model.add(Dense(2, activation='softmax', name='dense'))

# model summary
model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model fitting
model.fit_generator(train_generator,
                    steps_per_epoch=15,
                    epochs=16,
                    validation_data=valid_generator,
                    validation_steps=15)

# model evaluation
model.evaluate_generator(valid_generator)
