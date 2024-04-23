print("="*10, "SCRIPT START", "="*10)
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from keras import layers


# import dataset
data = keras.datasets.cifar10
# training and testing subset
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print(train_images[7])

# Pre-process images
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()
plt.savefig('./lab2/test_image.png')
train_images = train_images/255.0
test_images = test_images/255.0

# Create model
model = keras.Sequential()
model.add(keras.Input(shape=(32, 32, 3)))
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Flatten())
model.add(keras.layers.Dense(256, activation="sigmoid"))
model.add(keras.layers.Dense(124, activation="sigmoid"))
model.add(keras.layers.Dense(100, activation="softmax"))

# Combine model
model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"]
)

# train model
model.fit(train_images, train_labels, epochs=20)

# model prediction
prediction = model.predict(test_images)

# print("Prediction: ", prediction)
print("Prediction:", class_names[np.argmax(prediction[0])])

# visualization
for i in range(20, 30):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i][0]])
    plt.ylabel("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.savefig('./lab2/prediction_img_{}.png'.format(i))
    plt.show()
print("="*10, "SCRIPT END", "="*10)