#coding:utf-8

import os
from PIL import Image
import numpy as np

trainPath = r"E:\Share\AutoML_8成準確率資料\train32\\"
testPath = r"E:\Share\AutoML_8成準確率資料\test32\\"


# 資料夾內照片種類以及張數
def showPicNumber(dirName, label):

    labelset = set(label)
    labels = list(label)

    for x in labelset:
        pass
        print(dirName ," 標籤 ", x," ", labels.count(x))
#讀取訓練以及測試資料夾圖片，圖片為灰階圖，1通道
def load_data():
    imgs_train = os.listdir(trainPath)#列出訓練資料檔案夾內所有檔案名稱
    imgs_test = os.listdir(testPath)#列出測試資料檔案夾內所有檔案名稱
    firstImage = Image.open(trainPath + imgs_train[0])#開啟第一張圖片
    pixel = firstImage.size[0]#照片尺寸大小
    num_train = len(imgs_train)#訓練檔案夾有多少圖片
    num_test = len(imgs_test)#訓練檔案夾有多少圖片


    data_1 = np.empty((num_train,1,pixel,pixel),dtype="float32")
    label_1 = np.empty((num_train,),dtype="uint8")
    data_2 = np.empty((num_test,1,pixel,pixel),dtype="float32")
    label_2 = np.empty((num_test,),dtype="uint8")
    
    for i in range(num_train):
        img_train = Image.open(trainPath + imgs_train[i])
        arr_1 = np.asarray(img_train,dtype="float32")
        data_1[i,:,:,:] = arr_1
        label_1[i] = int(imgs_train[i].split('.')[0])

        # 調整label

        if label_1[i] == 3:
            label_1[i] = 0
        else:
            label_1[i] = 1
        
    for i in range(num_test):
        img_test = Image.open(testPath + imgs_test[i])
        arr_2 = np.asarray(img_test,dtype="float32")
        data_2[i,:,:,:] = arr_2
        label_2[i] = int(imgs_test[i].split('.')[0])

        # 調整label

        if label_2[i] == 3:
            label_2[i] = 0
        else:
            label_2[i] = 1
    
    showPicNumber("訓練資料夾",label_1)
    showPicNumber("測試資料夾",label_2)

    return (data_1,label_1), (data_2,label_2)