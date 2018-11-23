import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout

num_classes = 10

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

try:
    model.load_weights("mnist.h5")
    print("success")
except:
    print("error")
	
def plot_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image,cmap='binary')
    plt.show()
    
img=np.array(Image.open('test.jpg').convert('L'))
plot_image(img)

x_Test = img.reshape(1,784).astype('float32')
x_Test_normalize = x_Test.astype('float32') / 255.0
prediction=model.predict_classes(x_Test)
print(prediction[0])

prediction=model.predict(x_Test)
print(prediction[0])

prediction=model.predict_classes(x_Test_normalize)
print(prediction[0])

prediction=model.predict(x_Test_normalize)
print(prediction[0])