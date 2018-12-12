
# coding: utf-8

# Import Library
from keras.utils import np_utils
import numpy as np
from six.moves import range
import random
from data1210 import load_data

dim = 3 # 1: grey
num_classes = 2
width_image = 512
height_image = 512

np.random.seed(10)

#打亂數據
(x_Train, y_Train), (x_Test, y_Test) = load_data() # (image, label)
index_1 = [i for i in range(len(x_Train))]
random.shuffle(index_1)
x_Train = x_Train[index_1]
y_Train = y_Train[index_1]
print(x_Train.shape[0])
print(x_Test.shape[0])

# 資料前處理
x_Train4D = x_Train.reshape(x_Train.shape[0],width_image,height_image,dim).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0],width_image,height_image,dim).astype('float32')
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

# 建立模型
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=(width_image,height_image,dim),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

print(model.summary())

# 訓練模型
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) 
train_history=model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.1,epochs=12, batch_size=32,verbose=1)

import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

# 評估模型準確率
scores = model.evaluate(x_Test4D_normalize , y_TestOneHot)
scores[1]

# 預測結果
prediction=model.predict_classes(x_Test4D_normalize)
prediction[:num_classes]

# Save model
try:
    model.save_weights("mnist.h5")
    print("success")
except:
    print("error")

