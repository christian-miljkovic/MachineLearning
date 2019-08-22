# File to make the predicition of images
import tensorflow as tf
import numpy as np 
from tensorflow.keras.preprocessing import image

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model.h5')

# Configure model for categorical classification
model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

# Process the image into a proper value that the model can predict
dog_img = './data_sets/test_set/dogs/dog.4001.jpg'
cat_img = './data_sets/test_set/cats/cat.4990.jpg'

predict_img = image.load_img(cat_img, target_size = (64,64))
predict_img = image.img_to_array(predict_img)
predict_img = np.expand_dims(predict_img, axis = 0)

# Predict the image value
result = model.predict(predict_img)
if result[0][0] >= 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print(result[0][0])
print(prediction)






