# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:58:51 2018
使用 ImageDataGenerator
@author: mars0925
"""

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D #Convolution layer
from keras.layers import MaxPooling2D #Pooling layer
from keras.layers import Flatten ##扁平化
from keras.layers import Dense #神經網路層
from keras.layers import Dropout

trainPath = r"D:\AI\秀傳提供的bonescan檔案\已處理\2_dataset128_ImageDataGenerator\train"#訓練集路徑
testPath = r"D:\AI\秀傳提供的bonescan檔案\已處理\2_dataset128_ImageDataGenerator\test"#訓練集路徑
img_size = 128#照片大小
batch_size = 32#一次要學習幾張
trainSample = 800#訓練習樣本數
testSample = 190#測試集樣本數
epochs = 20 #訓練回數



# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
#添加捲積層
#filter 特徵探測器個數
#kernel_size 探測器的大小
#activation= "relu" 激活函數
#input_shape = (64, 64, 3) 輸入的圖片大小,第一層需要設定
classifier.add(Convolution2D(filters= 32, kernel_size=(3,3), activation= "relu", input_shape = (img_size, img_size, 1),padding='same'))
classifier.add(Dropout(rate=0.3))

# Step 2 - Pooling
#添加池化層
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#第二層
classifier.add(Convolution2D(filters= 32, kernel_size=(3,3), activation = 'relu',padding='same'))
classifier.add(Dropout(rate=0.3))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
#扁平化
classifier.add(Flatten())

# Step 4 - Full connection
#連結到神經網路
#units=128神經元個數
classifier.add(Dense(units=32, activation = 'relu'))#隱藏層
classifier.add(Dense(units=1, activation = 'sigmoid'))#輸出層


print(classifier.summary())

try:
    classifier.load_weights("./xrayCnnModel.h5")
    print("載入模型成功!繼續訓練模型")
except Exception as e:
    print("出錯了", e)   
    print("載入模型失敗!開始訓練一個新模型")

# Compiling the CNN
# optimizer = 'adam' 優化算法
# loss = 'binary_crossentropy' 損失函數
# 要依據分類個數區分 如果是多類別要設定為categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# 對圖像進行預先處理
# 使用keras來產生較多的train image
#rescale=1./255 像素除以255 特徵縮放
#shear_range=0.2 #圖片轉向
#zoom_range=0.2 #圖片縮放
#horizontal_flip=True #圖片水平翻轉
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2,horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)#測試集只需要特徵縮放處理即可

#target_size = (64, 64) 圖片大小
#batch_size = 32每次生成的張數
# class_mode = 'binary' 分類的種類 如果是多類別 設定categorical
#color_mode: "grayscale", "rbg" 之一。默认："rgb"。图像是否被转换成 1 或 3 个颜色通道。
training_set = train_datagen.flow_from_directory(trainPath,target_size = (img_size, img_size),batch_size = batch_size,class_mode = 'binary',color_mode = "grayscale")

test_set = test_datagen.flow_from_directory(testPath,target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'binary',color_mode = "grayscale")

#用生成的圖片訓練模型
#steps_per_epoch=250,它通常應等於數據集的樣本數除以bathc_size 8000/32
#epochs=25,期數
#validation_data = test_set,測試集
#validation_steps = 62.5 測試集個數/batch_size 2000/32

train_history = classifier.fit_generator(training_set, steps_per_epoch=trainSample/batch_size, epochs=epochs,validation_data = test_set,validation_steps = testSample/batch_size)

classifier.save_weights("./xrayCnnModel.h5")

from tool import show_train_history

#劃出訓練圖形
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

