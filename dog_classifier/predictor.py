# File to make the predicition of images
import tensorflow as tf
import numpy as np 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def make_prediction(img):

    # Recreate the exact same model, including weights and optimizer.
    model = tf.keras.models.load_model('./my_model.h5')

    predict_img = image.load_img(img, target_size = (64,64))
    predict_img = image.img_to_array(predict_img)
    predict_img = np.expand_dims(predict_img, axis = 0)

    # Predict the image value
    result = model.predict(predict_img)
    if result[0][0] >= 0.5:
        prediction = 'dog'
    else:
        prediction = 'cat'

    print(result[0][0])
    return 'This image is of a ' + str(prediction) +' with an accuracy of: ' + str(result[0][0])

