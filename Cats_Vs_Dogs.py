import numpy as np
import tensorflow as tf
import pickle

CATEGORIES = ['Cat','Dog']

#Loading in Data
#Training DATA
pickle_in = open("X.pickle","rb")
X = np.array(pickle.load(pickle_in))
pickle_in = open("y.pickle","rb")
y = np.array(pickle.load(pickle_in))

#Testing DATA
pickle_in = open("X_test.pickle","rb")
X_test = np.array(pickle.load(pickle_in))
pickle_in = open("y_test.pickle","rb")
y_test = np.array(pickle.load(pickle_in))



#Building the model

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(80,80)),
   tf.keras.layers.Dense(128,activation='relu'),
   tf.keras.layers.Dense(2)])


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

'''
model.fit(X,y,epochs=10)
model.save("D:\Programming\Machine Learning Kaggle")
'''

model = tf.keras.models.load_model("D:\Programming\Machine Learning Kaggle\Project_Model")
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(X_test)
counter = 0
for i in range(len(X_test)):
    if np.argmax(predictions[i]) != y_test[i]:
        counter += 1
        
accuracy = 1 - ((counter/len(X_test)))
print(str(100*accuracy) + " %")