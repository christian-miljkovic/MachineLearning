# file to run the convolution neural net
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Initializing sequential model
model = Sequential()

# Add a convolutional layer using an activation function ReLu
model.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Add a max pooling layer to the model
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten the tensors -> this will act as the input layer of a futuer neural net
model.add(Flatten())

# Connect the neural net
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'softmax'))

# Compile the neural net using an Adam optimizer, binary cross entropy loss function, and accurcay as success metric
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Do image pre-processing and read-in the data
training_data = ImageDataGenerator(
    rescale=1/255,
    shear_range =0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_data = ImageDataGenerator(rescale=1/255)

training_set  = training_data.flow_from_directory(
    'data_sets/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_data.flow_from_directory(
    'data_sets/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Configure model for categorical classification
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

# Fit the model to the data
model.fit_generator(
    training_set,
    steps_per_epoch=1000,
    epochs=10,
    validation_data=test_set,
    validation_steps=100
)

# Save entire model to a HDF5 file
model.save('my_model.h5')