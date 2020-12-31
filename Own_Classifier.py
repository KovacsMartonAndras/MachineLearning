import numpy as np 
import cv2
import os 
import random
import pickle
import matplotlib.pyplot as plt


DATADIR_TRAINING = "D:\\Programming\\Machine Learning Kaggle\\Cats_vs_Dogs_datasets\\dataset\\training_set"
DATADIR_TEST = "D:\\Programming\\Machine Learning Kaggle\\Cats_vs_Dogs_datasets\\dataset\\test_set"
CATEGORIES = ["cats","dogs"]
IMG_SIZE= 80

def create_training_data(DataDir):
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DataDir,category)
        class_num = CATEGORIES.index(category)
        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
                re_array = cv2.resize(image_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([re_array,class_num])
            except Exception as e:
                pass
    return np.array(training_data)
       
        
training_data = create_training_data(DATADIR_TRAINING)

random.shuffle(training_data)

test_data = create_training_data(DATADIR_TEST)
random.shuffle(test_data)

#Create X and y 
def reshape_X_and_y(dataset):
    X = []
    y = []
    for features, label in dataset:
        X.append(features/255)
        y.append(label)
    X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    return X,y

X,y = reshape_X_and_y(training_data)
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


#testing data
X,y = reshape_X_and_y(test_data)
print("Testing Data X shape: " + str(X.shape))
print("Testing Data y shape: " + str(np.array(y).shape))
pickle_out = open("X_test.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()




