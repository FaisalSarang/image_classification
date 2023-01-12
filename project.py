import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt


tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)

#Shapes of imported arrays
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train.shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

#Encoding labels
y_train_encoded = to_categorical(y_train)
y_test_encoded  = to_categorical(y_test)

#Unrolling N dimensional arrays to vectors
x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

#Data normalisation
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)
epsilon = 1e-10
x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

#Creating the model
model = Sequential([
    Dense(128, activation = 'relu', input_shape = (784, )),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

#Compiling the model
model.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#Training the model
model.fit(x_train_norm, y_train_encoded, epochs = 3)

#Evaluating the model
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy;', accuracy*100)

#Predictions
preds = model.predict(x_test_norm)
plt.figure(figsize = (12, 12))
start_index = 0

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+i])
    gt = y_test[start_index+i]
    
    col = 'g'
    if pred != gt:
        col = 'r'
    
    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i, pred, gt), color = col)
    plt.imshow(x_test[start_index+i], cmap = 'binary')
    
plt.show()