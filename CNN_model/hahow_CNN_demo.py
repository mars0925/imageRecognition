import numpy as np
import pandas as pd
import keras
# from keras.utils import np_utils
from testpic import load_data
#準備資料

#載入手寫辨識資料 分成訓練以及測試的資料集
(x_train, y_train) , (x_test, y_test) = load_data()

#訓練和測試資料分別有多少
# print("x_train_image:", x_train.shape)
# print("x_train_label:", y_train.shape)

# print("x_TEST_image:", x_test.shape)
# print("x_TEST_label:", y_test.shape)

# print(x_train.shape[0])

# print(x_train[0])#第一章圖形的陣列


import matplotlib.pyplot as plt

#將資料矩陣轉成圖片的Function
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap = "binary")
    plt.show()
print(x_train[0].shape)
plot_image(x_train[0])#畫資料中第一張圖


#多增加一個顏色的維度
# x_train4D = x_train.reshape(x_train.shape[0],28,28,1).astype("float32")
# x_test4D = x_test.reshape(x_test.shape[0],28,28,1).astype("float32")

# print(x_train4D.shape)

#將數值縮小到0~1 灰階的圖片是0 ~255之間
x_train4D_normalize = x_train / 255.0
x_test4D_normalize = x_test / 255.0

#把類別作one hot encoding
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

# print(y_trainOneHot)

#建立CNN模型
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

models = Sequential()

#fitter =  16  kernel size = (5,5) padding = same
# fitter =  16 隨機創造多少個dector
# kernel size dector的大小要多大
# padding 超出邊界的時候值要怎麼補
# input_shape 告訴model圖片大小
# activation 激活函數

models.add(Conv2D(filters = 16, kernel_size = (5,5), padding = "same", input_shape = (28, 28, 1),activation = "relu"))

#maxPooling size = (2,2)

models.add(MaxPool2D(pool_size = (2,2)))


models.add(Conv2D(filters = 36, kernel_size = (5,5), padding = "same", input_shape = (28, 28, 1),activation = "relu"))

models.add(MaxPool2D(pool_size = (2,2)))

#Drop部分神經元 避免overfitting
models.add(Dropout(0.25))
#平坦化
models.add(Flatten())
#深度學習的神經網路
models.add(Dense(128, activation = "relu"))
models.add(Dropout(0.5))

#softmax 當成預測類別的函數
models.add(Dense(10, activation = "softmax"))

print(models.summary())


#訓練模型
models.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

train_history = models.fit(x = x_train4D_normalize, y = y_trainOneHot, validation_split = 0.2, epochs =20, batch_size = 300, verbose = 2 )


