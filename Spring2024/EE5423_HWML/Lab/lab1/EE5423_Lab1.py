print("="*10, "SCRIPT START", "="*10)
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

# import dataset
data = keras.datasets.fashion_mnist
# training and testing subset
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = [
    'T-shirt/top', 
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

print(train_images[7])

# Pre-process images
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()
plt.savefig('./test_image.png')
train_images = train_images/255.0
test_images = test_images/255.0

# Create model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # First layer to flaten the image to array
    # keras.layers.Dense(128, activation="sigmoid"), # Sencond layer
    # keras.layers.Dense(64, activation="sigmoid"), # Sencond layer
    keras.layers.Dense(10, activation="softmax") # Output layer
])

# Combine model
model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"]
)

# train model
model.fit(train_images, train_labels, epochs=5)

# model prediction
prediction = model.predict(test_images)

# print("Prediction: ", prediction)
print("Prediction:", class_names[np.argmax(prediction[0])])

# visualization
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.ylabel("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.savefig('./prediction_img_{}.png'.format(i))
    plt.show()
print("="*10, "SCRIPT END", "="*10)