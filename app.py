# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.preprocessing.image import img_to_array
# import base64
# from PIL import Image
# import io
# import matplotlib.pyplot as plt

# categories = ['dog', 'cat']


# train = ImageDataGenerator(rescale=1/255)
# validation = ImageDataGenerator(rescale=1/255)

# train_dataset = train.flow_from_directory('cats_dogs_data/training_set',
#                                         target_size=(100, 100),
#                                         #batch_size=5,
#                                         class_mode='binary')

# validation_dataset = train.flow_from_directory('cats_dogs_data/test_set',
#                                         target_size=(100, 100),
#                                         #batch_size=5,
#                                         class_mode='binary')


# model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),

#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(64, activation='relu'),

#         tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# print(validation_dataset.class_indices)

# model.compile(loss='binary_crossentropy',
#                 optimizer='adam',
#                 metrics=['accuracy'])

# model.fit(train_dataset,
#         #steps_per_epoch=800,
#         epochs=25,
#         validation_data=validation_dataset
#         )

# model.save("asl_model.h5")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import img_to_array
import base64
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model("asl_model.h5")

from flask import request
from flask import jsonify
from flask import Flask
from flask import render_template


app = Flask(__name__)

def get_model():
    global model
    model = keras.models.load_model("asl_model.h5")
    print('model loaded')

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.vstack([image])

    return image

@app.route("/")
def hello():
    return render_template('index.html')

get_model()

@app.route("/predict", methods=['POST'])
def predict():

    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(100, 100))

    prediction = model.predict(processed_image)
    print(prediction)
    
    animal = ''
    if prediction == 0:
            animal = 'cat'
    else:
            animal = 'dog'

    response={
        'name': animal
    }

    print('prediction: ' + animal)
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)