import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels), (test_images,test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

             
#Plotting the images
#plt.figure()
#plt.imshow(train_images[2])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#Scaling the values between 0 and 1
train_images = train_images / 255.0

test_images = test_images / 255.0

#ModelBuilding

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                             tf.keras.layers.Dense(128,activation='relu'),
                             tf.keras.layers.Dense(10)])

#Model is made out of 3 layers, The 1st one to Flatten the values, 2nd one is 
#a hidden layer, the 3rd one is the output

#Model compile step
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#TRAINING
model.fit(train_images, train_labels, epochs=10)


#Test accuracy
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
plt.figure()
plt.imshow(test_images[201])
plt.colorbar()
plt.grid(False)
plt.show()
print(class_names[np.argmax(predictions[201])])