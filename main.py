from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape[0])
x_train2 = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])